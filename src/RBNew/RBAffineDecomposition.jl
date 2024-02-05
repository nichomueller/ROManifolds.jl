struct ReducedIntegrationDomain{Ms,Mt}
  measure_space::Ms
  measure_time::Mt
end

function get_mdeim_indices(A::AbstractMatrix{T}) where T
  m,n = size(A)
  proj = zeros(T,m)
  res = zeros(T,m)
  idx = zeros(Int,n)
  idx[1] = argmax(abs.(A[:,1]))

  if n > 1
    @inbounds for i = 2:n
      @. proj = A[:,1:i-1]*(A[idx[1:i-1],1:i-1] \ A[idx[1:i-1],i])
      @. res = A[:,i] - proj
      idx[i] = argmax(abs.(res))
    end
  end

  return idx
end

function reduce_triangulation(test,trian,snaps,indices_space)
  cell_dof_ids = get_cell_dof_ids(test,trian)
  recast_indices_space = recast_indices(snaps,indices_space)
  red_integr_cells = get_reduced_cells(recast_indices_space,cell_dof_ids)
  red_trian = view(trian,red_integr_cells)
  return red_trian
end

function project_basis_space(basis_space::AbstractMatrix,test::RBSpace)
  compress(get_basis_space(test),basis_space)
end

function project_basis_space(basis_space::AbstractMatrix,test::RBSpace,trial::RBSpace)
  compress(get_basis_space(test),get_basis_space(trial),basis_space)
end

function combine_basis_time(test::RBSpace;kwargs...)
  get_basis_time(test)
end

function combine_basis_time(
  test::RBSpace,
  trial::RBSpace;
  combine=(x,y)->x)

  test_basis = get_basis_time(test)
  trial_basis = get_basis_time(trial)
  time_ndofs = size(test_basis,1)
  nt_test = size(test_basis,2)
  nt_trial = size(trial_basis,2)

  bt_proj = zeros(T,time_ndofs,nt_test,nt_trial)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_trial, it = 1:nt_test
    bt_proj[:,it,jt] .= test_basis[:,it].*trial_basis[:,jt]
    bt_proj_shift[2:end,it,jt] .= test_basis[2:end,it].*trial_basis[1:end-1,jt]
  end

  combine(bt_proj,bt_proj_shift)
end

struct AffineDecomposition{T,N,I}
  basis_space::AbstractMatrix{T}
  basis_time::AbstractArray{T,N}
  mdeim_interpolation::LU
  integration_domain::I
end

struct TrivialAffineDecomposition{T,N}
  projection::AbstractArray{T,N}
end

function reduced_form()
  basis_space,basis_time = compress(snaps;kwargs...)
  indices_space = get_mdeim_indices(basis_space)
  interp_basis_space = view(basis_space,indices_space,:)
  if rbinfo.st_mdeim
    indices_time = get_mdeim_indices(basis_time)
    interp_basis_time = view(basis_time,indices_time,:)
    interp_basis_space_time = LinearAlgebra.kron(interp_basis_time,interp_basis_space)
    lu_interp = lu(interp_basis_space_time)
  else
    indices_time = eachindex(times)
    lu_interp = lu(interp_bs)
  end
  red_trian = reduce_triangulation(test,trian,snaps,indices_space)
  proj_basis_space,proj_basis_time = project_space_time(basis_space,basis_time,args...;kwargs...)
  time_domain = ReducedTimeDomain(indices_time)
  AffineDecomposition(proj_basis_space,proj_basis_time,lu_interp,time_domain)
end

function reduced_bilinear_form()
  snaps = collect_matrix_snapshots()
  reduced_form()
end

function reduced_linear_form()
  snaps = collect_vector_snapshots()
  reduced_form()
end

function reduced_bilinear_linear_form()
  snaps = collect_matrix_vector_snapshots()
  reduced_form()
end

function mdeim_solve!(cache,a::AffineDecomposition,b::AbstractArray)
  cache_solve,cache_recast = cache

  mdeim_interpolation = a.mdeim_interpolation
  setsize!(cache_solve,size(b))
  coeff = cache_solve.array
  ldiv!(coeff,mdeim_interpolation,b)

  recast_coefficient!(cache_recast,a,coeff)
end

function recast_coefficient!(cache_recast,a::AffineDecomposition,coeff::AbstractMatrix)
  Nt = num_times(a)
  ns = num_reduced_space_dofs(a)
  setsize!(cache_recast,(Nt,ns))
  array = get_array(cache_recast)

  @inbounds for n = eachindex(array)
    array[n] = coeff[:,(n-1)*Nt+1:n*Nt]'
  end

  ParamArray(array)
end

function recast_coefficient!(cache_recast,a::AffineDecomposition,coeff::AbstractVector)
  Nt = num_times(a)
  nt = num_reduced_times(a)
  ns = num_reduced_space_dofs(a)
  basis_time = get_basis_time(a)

  setsize!(cache_recast,(Nt,ns))
  array = get_array(cache)

  for j in 1:ns
    sorted_idx = [(i-1)*ns+j for i = 1:nt]
    @inbounds for n = eachindex(array)
      array[n][:,j] = basis_time*coeff[sorted_idx,n]
    end
  end

  ParamArray(array)
end

# cache for mdeim:
# 1) compute reduced matrices and vectors on the reduced integration domain
# 2) execute the mdeim_solve!, which includes the solve! and the recast_coefficient!
# 3) sum of the reduced contributions (i.e. the Kronecker products)

abstract type RBContributionMap <: Map end
struct RBVecContributionMap <: RBContributionMap end
struct RBMatContributionMap <: RBContributionMap end

function Arrays.return_cache(::RBVecContributionMap,snaps::ParamArray{Vector{T}}) where T
  array_coeff = zeros(T,1)
  array_proj = zeros(T,1)
  CachedArray(array_coeff),CachedArray(array_proj),CachedArray(array_proj)
end

function Arrays.return_cache(::RBMatContributionMap,snaps::ParamArray{Vector{T}}) where T
  array_coeff = zeros(T,1,1)
  array_proj = zeros(T,1,1)
  CachedArray(array_coeff),CachedArray(array_proj),CachedArray(array_proj)
end

function Arrays.evaluate!(
  ::RBVecContributionMap,
  cache,
  proj_basis_space::AbstractVector,
  basis_time::Matrix{T},
  coeff::Matrix{T}) where T

  @assert length(proj_basis_space) == size(coeff,2)
  proj1 = testitem(proj_basis_space)

  cache_coeff,cache_proj,cache_proj_global = cache
  num_rb_times = size(basis_time,2)
  num_rb_dofs = length(proj1)*size(basis_time,2)
  setsize!(cache_coeff,(num_rb_times,))
  setsize!(cache_proj,(num_rb_dofs,))
  setsize!(cache_proj_global,(num_rb_dofs,))

  array_coeff = cache_coeff.array
  array_proj = cache_proj.array
  array_proj_global = cache_proj_global.array
  array_proj_global .= zero(T)

  @inbounds for i = axes(coeff,2)
    array_coeff .= basis_time'*coeff[:,i]
    LinearAlgebra.kron!(array_proj,proj_basis_space[i],array_coeff)
    array_proj_global .+= array_proj
  end

  array_proj_global
end

function Arrays.evaluate!(
  ::RBMatContributionMap,
  cache,
  proj_basis_space::AbstractVector,
  basis_time::Array{T,3},
  coeff::Matrix{T}) where T

  @assert length(proj_basis_space) == size(coeff,2)
  proj1 = testitem(proj_basis_space)

  cache_coeff,cache_proj,cache_proj_global = cache
  num_rb_times_row = size(basis_time,2)
  num_rb_times_col = size(basis_time,3)
  num_rb_rows = size(proj1,1)*size(basis_time,2)
  num_rb_cols = size(proj1,2)*size(basis_time,3)
  setsize!(cache_coeff,(num_rb_times_row,num_rb_times_col))
  setsize!(cache_proj,(num_rb_rows,num_rb_cols))
  setsize!(cache_proj_global,(num_rb_rows,num_rb_cols))

  array_coeff = cache_coeff.array
  array_proj = cache_proj.array
  array_proj_global = cache_proj_global.array
  array_proj_global .= zero(T)

  for i = axes(coeff,2)
    for col in 1:num_rb_times_col
      for row in 1:num_rb_times_row
        @fastmath @inbounds array_coeff[row,col] = sum(basis_time[:,row,col].*coeff[:,i])
      end
    end
    LinearAlgebra.kron!(array_proj,proj_basis_space[i],array_coeff)
    array_proj_global .+= array_proj
  end

  array_proj_global
end

function rb_contribution!(
  cache,
  k::RBContributionMap,
  a::AffineDecomposition,
  coeff::ParamArray)

  basis_space_proj = a.basis_space
  basis_time = last(a.basis_time)
  map(coeff) do cn
    copy(evaluate!(k,cache,basis_space_proj,basis_time,cn))
  end
end

function rb_contribution!(
  cache,
  k::RBContributionMap,
  a::TrivialAffineDecomposition,
  coeff::ParamArray)

  array = [a.projection for _ = eachindex(coeff)]
  ParamArray(array)
end

function zero_rb_contribution(
  ::RBVecContributionMap,
  rbinfo::RBInfo,
  rbspace::RBSpace{T}) where T

  nrow = num_rb_ndofs(rbspace)
  [zeros(T,nrow) for _ = 1:rbinfo.nsnaps_test]
end
