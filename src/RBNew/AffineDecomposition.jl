# OFFLINE PHASE

function get_mdeim_indices(A::AbstractMatrix{T}) where T
  m,n = size(A)
  proj = zeros(T,m)
  res = zeros(T,m)
  I = zeros(Int,n)
  I[1] = argmax(abs.(A[:,1]))

  if n > 1
    @inbounds for i = 2:n
      Bi = view(A,:,1:i-1)
      Ci = view(A,I[1:i-1],1:i-1)
      Di = view(A,I[1:i-1],i)
      proj .= Bi*(Ci \ Di)
      res .= A[:,i] - proj
      I[i] = argmax(abs.(res))
    end
  end

  return I
end

function recast_indices_space(a::AbstractArray,indices_space)
  return indices_space
end

function recast_indices_space(a::NnzSnapshots,indices_space)
  aitem = first(a.values)
  rows,cols, = findnz(aitem)
  rc = (cols .- 1)*aitem.m .+ rows
  return rc[indices_space]
end

function vector_to_matrix_indices(vec_indices,nrows)
  icol = slow_index(vec_indices,nrows)
  irow = fast_index(vec_indices,nrows)
  return irow,icol
end

function get_reduced_cells(idx::AbstractVector{T},cell_dof_ids) where T
  cells = T[]
  for cell = eachindex(cell_dof_ids)
    dofs = cell_dof_ids[cell]
    if !isempty(intersect(idx,dofs))
      append!(cells,cell)
    end
  end
  return unique(cells)
end

function reduce_triangulation(
  op::RBOperator,
  trian::Triangulation,
  indices_space::AbstractVector)

  test = get_test(op)
  cell_dof_ids = get_cell_dof_ids(test,trian)
  indices_space_rows = slow_index(indices_space,num_free_dofs(test))
  red_integr_cells = get_reduced_cells(indices_space_rows,cell_dof_ids)
  red_trian = view(trian,red_integr_cells)
  return red_trian
end

function project_basis_space(A::AbstractMatrix,test::RBSpace)
  basis_test = get_basis_space(test)
  map(A) do a
    basis_test'*a
  end
end

function project_basis_space(A::AbstractMatrix,trial::RBSpace,test::RBSpace)
  basis_test = get_basis_space(test)
  basis_trial = get_basis_space(trial)
  map(A.values) do A
    basis_test'*A*basis_trial
  end
end

function combine_basis_time(test::RBSpace;kwargs...)
  get_basis_time(test)
end

function combine_basis_time(
  trial::RBSpace,
  test::RBSpace;
  combine=(x,y)->x)

  test_basis = get_basis_time(test)
  trial_basis = get_basis_time(trial)
  time_ndofs = size(test_basis,1)
  nt_test = size(test_basis,2)
  nt_trial = size(trial_basis,2)

  T = eltype(get_vector_type(test))
  bt_proj = zeros(T,time_ndofs,nt_test,nt_trial)
  bt_proj_shift = copy(bt_proj)
  @inbounds for jt = 1:nt_trial, it = 1:nt_test
    bt_proj[:,it,jt] .= test_basis[:,it].*trial_basis[:,jt]
    bt_proj_shift[2:end,it,jt] .= test_basis[2:end,it].*trial_basis[1:end-1,jt]
  end

  combine(bt_proj,bt_proj_shift)
end

struct ReducedIntegrationDomain{T}
  indices_space::AbstractVector{T}
  indices_time::AbstractVector{T}
end

get_indices_space(i::ReducedIntegrationDomain) = i.indices_space
get_indices_time(i::ReducedIntegrationDomain) = i.indices_time
get_all_indices_time(i::ReducedIntegrationDomain...) = union(map(get_indices_time,i))
get_common_indices_time(i::ReducedIntegrationDomain...) = intersect(map(get_indices_time,i))

struct AffineDecomposition{A,B,C}
  basis_space::A
  basis_time::B
  mdeim_interpolation::LU
  integration_domain::C
end

get_integration_domain(a::AffineDecomposition) = a.integration_domain
get_indices_space(a::AffineDecomposition) = get_indices_space(get_integration_domain(a))
get_indices_time(a::AffineDecomposition) = get_indices_time(get_integration_domain(a))
get_all_indices_time(a::AffineDecomposition...) = union(map(get_indices_time,a))

const AffineContribution = Contribution{AffineDecomposition}

affine_contribution() = Contribution(IdDict{Triangulation,AffineDecomposition}())

function mdeim(
  rbinfo::RBInfo,
  op::RBOperator,
  trian::Triangulation,
  basis_space::AbstractMatrix,
  basis_time::AbstractMatrix)

  indices_space = get_mdeim_indices(basis_space)
  interp_basis_space = view(basis_space,indices_space,:)
  if rbinfo.st_mdeim
    indices_time = get_mdeim_indices(basis_time)
    interp_basis_time = view(basis_time,indices_time,:)
    interp_basis_space_time = LinearAlgebra.kron(interp_basis_time,interp_basis_space)
    lu_interp = lu(interp_basis_space_time)
  else
    indices_time = axes(basis_time,1)
    lu_interp = lu(interp_basis_space)
  end
  rindices_space = recast_indices_space(basis_space,indices_space)
  red_trian = reduce_triangulation(op,trian,indices_space)
  integration_domain = ReducedIntegrationDomain(rindices_space,indices_time)
  return lu_interp,red_trian,integration_domain
end

function reduced_vector_form!(
  a::AffineContribution,
  rbinfo::RBInfo,
  op::RBOperator,
  s::AbstractTransientSnapshots,
  trian::Triangulation)

  test = op.test
  basis_space,basis_time = compute_bases(s;ϵ=get_tol(rbinfo))
  lu_interp,red_trian,integration_domain = mdeim(rbinfo,op,trian,basis_space,basis_time)
  proj_basis_space = project_basis_space(basis_space,test)
  comb_basis_time = combine_basis_time(test)
  a[red_trian] = AffineDecomposition(proj_basis_space,comb_basis_time,lu_interp,integration_domain)
  return a
end

function reduced_matrix_form!(
  a::AffineContribution,
  rbinfo::RBInfo,
  op::RBOperator,
  s::AbstractTransientSnapshots,
  trian::Triangulation;
  kwargs...)

  trial = op.trial
  test = op.test
  basis_space,basis_time = compute_bases(s;ϵ=get_tol(rbinfo))
  lu_interp,red_trian,integration_domain = mdeim(rbinfo,op,trian,basis_space,basis_time)
  proj_basis_space = project_basis_space(basis_space,trial,test)
  comb_basis_time = combine_basis_time(trial,test;kwargs...)
  a[red_trian] = AffineDecomposition(proj_basis_space,comb_basis_time,lu_interp,integration_domain)
  return a
end

function reduced_vector_form(
  solver::RBSolver,
  op::RBOperator,
  c::ArrayContribution)

  rbinfo = get_info(solver)
  a = affine_contribution()
  for (trian,values) in c.dict
    reduced_vector_form!(a,rbinfo,op,values,trian)
  end
  return a
end

function reduced_matrix_form(
  solver::RBSolver,
  op::RBOperator,
  c::ArrayContribution;
  kwargs...)

  rbinfo = get_info(solver)
  a = affine_contribution()
  for (trian,values) in c.dict
    reduced_matrix_form!(a,rbinfo,op,values,trian;kwargs...)
  end
  return a
end

function reduced_matrix_form(
  solver::RBThetaMethod,
  op::RBOperator,
  contribs::Tuple{Vararg{ArrayContribution}})

  fesolver = get_fe_solver(solver)
  θ = fesolver.θ
  map(enumerate(contribs)) do (i,c)
    combine = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*(x-y)
    reduced_matrix_form(solver,op,c;combine)
  end
end

function reduced_matrix_vector_form(
  solver::RBSolver,
  op::RBOperator,
  s::AbstractTransientSnapshots)

  contribs_mat,contribs_vec, = collect_matrices_vectors(solver,op,s)
  red_mat = reduced_matrix_form(solver,op,contribs_mat)
  red_vec = reduced_vector_form(solver,op,contribs_vec)
  return red_mat,red_vec
end

# ONLINE PHASE

function collect_matrices_vectors!(
  solver::ThetaRBSolver{Affine},
  op::RBOperator,
  amat::Tuple{Vararg{AffineDecomposition}},
  amat_t::Tuple{Vararg{AffineDecomposition}},
  avec::Tuple{Vararg{AffineDecomposition}},
  s::AbstractTransientSnapshots,
  cache)

  mat_tids = map(get_indices_time,amat)
  mat_t_tids = map(get_indices_time,amat_t)
  vec_tids = map(get_indices_time,avec)
  all_tids = get_all_indices_time(mat_tids...,mat_t_tids...,vec_tids...)
  s_tids = select_snapshots(s,:,all_tids)
  (smat,smat_t),svec = collect_matrices_vectors!(solver,op,s_tids,cache)
  red_smat = tensor_getindex(smat,get_indices_space(amat),indexin(all_tids,mat_tids),:)
  red_smat_t = tensor_getindex(smat_t,get_indices_space(amat_t),indexin(all_tids,mat_t_tids),:)
  red_svec = tensor_getindex(svec,get_indices_space(avec),indexin(all_tids,vec_tids),:)
  return red_smat,red_smat_t,red_svec
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

function zero_rb_contribution(
  ::RBVecContributionMap,
  rbinfo::RBInfo,
  rbspace::RBSpace{T}) where T

  nrow = num_rb_ndofs(rbspace)
  [zeros(T,nrow) for _ = 1:rbinfo.nsnaps_test]
end
