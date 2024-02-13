# OFFLINE PHASE

function get_mdeim_indices(A::AbstractMatrix{T}) where T
  m,n = size(A)
  proj = zeros(T,m)
  res = zeros(T,m)
  I = zeros(Int,n)
  I[1] = argmax(abs.(A[:,1]))

  if n > 1
    @inbounds for i = 2:n
      Bi = A[:,1:i-1]
      Ci = A[I[1:i-1],1:i-1]
      Di = A[I[1:i-1],i]
      proj .= Bi*(Ci \ Di)
      res .= A[:,i] - proj
      I[i] = argmax(abs.(res))
    end
  end

  return I
end

function get_mdeim_indices(A::NnzSnapshots)
  get_mdeim_indices(collect(A))
end

function recast_indices(A::AbstractMatrix,indices::AbstractVector{Int})
  return indices
end

function recast_indices(A::NnzSnapshots,indices::AbstractVector{Int})
  nonzero_indices = get_nonzero_indices(A)
  entire_indices = nonzero_indices[indices]
  return entire_indices
end

function get_reduced_cells(indices::AbstractVector{T},cell_dof_ids) where T
  cells = T[]
  for cell = eachindex(cell_dof_ids)
    dofs = cell_dof_ids[cell]
    if !isempty(intersect(indices,dofs))
      append!(cells,cell)
    end
  end
  return unique(cells)
end

function reduce_triangulation(
  op::RBOperator,
  trian::Triangulation,
  indices_space::AbstractVector)

  test = get_fe_test(op)
  cell_dof_ids = get_cell_dof_ids(test,trian)
  indices_space_rows = fast_index(indices_space,num_free_dofs(test))
  red_integr_cells = get_reduced_cells(indices_space_rows,cell_dof_ids)
  red_trian = view(trian,red_integr_cells)
  return red_trian
end

function compress_basis_space(A::AbstractMatrix,test::RBSpace)
  basis_test = get_basis_space(test)
  map(eachcol(A)) do a
    basis_test'*a
  end
end

function compress_basis_space(A::NnzSnapshots,trial::RBSpace,test::RBSpace)
  basis_test = get_basis_space(test)
  basis_trial = get_basis_space(trial)
  map(get_values(A)) do A
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
  indices_space::Vector{T}
  indices_time::Vector{T}
end

get_indices_space(i::ReducedIntegrationDomain) = i.indices_space
get_indices_time(i::ReducedIntegrationDomain) = i.indices_time
union_indices_time(i::ReducedIntegrationDomain...) = union(map(get_indices_time,i)...)

struct AffineDecomposition{M,A,B,C,D}
  mdeim_style::M
  basis_space::A
  basis_time::B
  mdeim_interpolation::LU
  integration_domain::C
  metadata::D
end

get_integration_domain(a::AffineDecomposition) = a.integration_domain
get_interp_matrix(a::AffineDecomposition) = a.mdeim_interpolation
get_indices_space(a::AffineDecomposition) = get_indices_space(get_integration_domain(a))
get_indices_time(a::AffineDecomposition) = get_indices_time(get_integration_domain(a))
num_space_dofs(a::AffineDecomposition) = @notimplemented
FEM.num_times(a::AffineDecomposition) = size(a.basis_time,1)
num_reduced_space_dofs(a::AffineDecomposition) = length(get_indices_space(a))
num_reduced_times(a::AffineDecomposition) = length(get_indices_time(a))

function _time_indices_and_interp_matrix(::SpaceTimeMDEIM,interp_basis_space,basis_time)
  indices_time = get_mdeim_indices(basis_time)
  interp_basis_time = view(basis_time,indices_time,:)
  interp_basis_space_time = LinearAlgebra.kron(interp_basis_time,interp_basis_space)
  lu_interp = lu(interp_basis_space_time)
  return indices_time,lu_interp
end

function _time_indices_and_interp_matrix(::SpaceOnlyMDEIM,interp_basis_space,basis_time)
  indices_time = axes(basis_time,1)
  lu_interp = lu(interp_basis_space)
  return indices_time,lu_interp
end

function mdeim(
  info::RBInfo,
  op::RBOperator,
  trian::Triangulation,
  basis_space::AbstractMatrix,
  basis_time::AbstractMatrix)

  indices_space = get_mdeim_indices(basis_space)
  interp_basis_space = view(basis_space,indices_space,:)
  indices_time,lu_interp = _time_indices_and_interp_matrix(info.mdeim_style,interp_basis_space,basis_time)
  recast_indices_space = recast_indices(basis_space,indices_space)
  red_trian = reduce_triangulation(op,trian,recast_indices_space)
  integration_domain = ReducedIntegrationDomain(recast_indices_space,indices_time)
  return lu_interp,red_trian,integration_domain
end

const AffineContribution = Contribution{AffineDecomposition}

affine_contribution() = Contribution(IdDict{Triangulation,AffineDecomposition}())

function reduced_vector_form!(
  a::AffineContribution,
  info::RBInfo,
  op::RBOperator,
  s::AbstractTransientSnapshots,
  trian::Triangulation)

  test = op.test
  basis_space,basis_time = reduced_basis(s;ϵ=get_tol(info))
  lu_interp,red_trian,integration_domain = mdeim(info,op,trian,basis_space,basis_time)
  proj_basis_space = compress_basis_space(basis_space,test)
  comb_basis_time = combine_basis_time(test)
  a[red_trian] = AffineDecomposition(
    info.mdeim_style,
    proj_basis_space,
    basis_time,
    lu_interp,
    integration_domain,
    comb_basis_time)
  return a
end

function reduced_matrix_form!(
  a::AffineContribution,
  info::RBInfo,
  op::RBOperator,
  s::AbstractTransientSnapshots,
  trian::Triangulation;
  kwargs...)

  trial = op.trial
  test = op.test
  basis_space,basis_time = reduced_basis(s;ϵ=get_tol(info))
  lu_interp,red_trian,integration_domain = mdeim(info,op,trian,basis_space,basis_time)
  proj_basis_space = compress_basis_space(basis_space,trial,test)
  comb_basis_time = combine_basis_time(trial,test;kwargs...)
  a[red_trian] = AffineDecomposition(
    info.mdeim_style,
    proj_basis_space,
    basis_time,
    lu_interp,
    integration_domain,
    comb_basis_time)
  return a
end

function reduced_vector_form(
  solver::RBSolver,
  op::RBOperator,
  c::ArrayContribution)

  info = get_info(solver)
  a = affine_contribution()
  for (trian,values) in c.dict
    reduced_vector_form!(a,info,op,values,trian)
  end
  return a
end

function reduced_matrix_form(
  solver::RBSolver,
  op::RBOperator,
  c::ArrayContribution;
  kwargs...)

  info = get_info(solver)
  a = affine_contribution()
  for (trian,values) in c.dict
    reduced_matrix_form!(a,info,op,values,trian;kwargs...)
  end
  return a
end

function reduced_matrix_form(
  solver::RBThetaMethod,
  op::RBOperator,
  contribs::Tuple{Vararg{ArrayContribution}})

  fesolver = get_fe_solver(solver)
  θ = fesolver.θ
  a = ()
  for (i,c) in enumerate(contribs)
    combine = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*(x-y)
    a = (a...,reduced_matrix_form(solver,op,c;combine))
  end
  return a
end

function reduced_matrix_vector_form(
  solver::RBSolver,
  op::RBOperator,
  s::AbstractTransientSnapshots)

  smdeim = select_snapshots(s,mdeim_params(solver.info))
  contribs_mat,contribs_vec = fe_matrix_and_vector(solver,op,smdeim)
  red_mat = reduced_matrix_form(solver,op,contribs_mat)
  red_vec = reduced_vector_form(solver,op,contribs_vec)
  return red_mat,red_vec
end

# ONLINE PHASE

function Algebra.allocate_matrix(::Type{V},m::Integer,n::Integer) where V
  T = eltype(V)
  zeros(T,m,n)
end

function allocate_coeff_matrix(
  a::AffineDecomposition{SpaceOnlyMDEIM},
  r::AbstractParamRealization)

  T = eltype(get_interp_matrix(a).factors)
  m = num_reduced_space_dofs(a)
  n = num_params(r)*num_times(a)
  allocate_matrix(Vector{T},m,n)
end

function allocate_coeff_matrix(
  a::AffineDecomposition{SpaceTimeMDEIM},
  r::AbstractParamRealization)

  T = eltype(get_interp_matrix(a).factors)
  m = num_reduced_space_dofs(a)*num_reduced_times(a)
  n = num_params(r)
  allocate_matrix(Vector{T},m,n)
end

function allocate_param_coeff_matrix(
  a::AffineDecomposition,
  r::AbstractParamRealization)

  T = eltype(get_interp_matrix(a).factors)
  m = num_times(a)
  n = num_reduced_space_dofs(a)
  mat = allocate_matrix(Vector{T},m,n)
  allocate_param_array(mat,num_params(r))
end

function allocate_mdeim_coeff(a::AffineDecomposition,r::AbstractParamRealization)
  cache_solve = allocate_coeff_matrix(a,r)
  cache_recast = allocate_param_coeff_matrix(a,r)
  return cache_solve,cache_recast
end

function allocate_mdeim_coeff(a::AffineContribution,r::AbstractParamRealization)
  cache_solve = array_contribution()
  cache_recast = array_contribution()
  for (trian,values) in a.dict
    cs,cr = allocate_mdeim_coeff(values,r)
    cache_solve[trian] = cs
    cache_recast[trian] = cr
  end
  return cache_solve,cache_recast
end

function mdeim_coeff!(
  cache,
  a::AffineDecomposition{SpaceOnlyMDEIM},
  b::AbstractMatrix)

  coeff,coeff_recast = cache
  mdeim_interpolation = a.mdeim_interpolation
  ldiv!(coeff,mdeim_interpolation,b)
  nt = num_times(a)
  @inbounds for i = eachindex(coeff_recast)
    coeff_recast[i] = transpose(coeff[:,(i-1)*nt+1:i*nt])
  end
end

function mdeim_coeff!(
  cache,
  a::AffineDecomposition{SpaceTimeMDEIM},
  b::AbstractMatrix)

  coeff,coeff_recast = cache
  mdeim_interpolation = a.mdeim_interpolation
  ns = num_reduced_space_dofs(a)
  nt = num_reduced_times(a)
  np = length(coeff_recast)

  bvec = reshape(b,:,np)
  ldiv!(coeff,mdeim_interpolation,bvec)
  for j in 1:ns
    sorted_idx = [(i-1)*ns+j for i = 1:nt]
    @inbounds for i = eachindex(coeff_recast)
      coeff_recast[i][:,j] = a.basis_time*coeff[sorted_idx,i]
    end
  end
end

function mdeim_coeff!(
  cache,
  a::AffineContribution,
  b::ArrayContribution)

  coeff,coeff_recast = cache
  for (trian,atrian) in a.dict
    cache_trian = coeff[trian],coeff_recast[trian]
    btrian = b[trian]
    mdeim_coeff!(cache_trian,atrian,btrian)
  end
  return coeff_recast
end

function allocate_mdeim_lincomb(
  test::RBSpace,
  r::AbstractParamRealization)

  V = get_vector_type(test)
  ns_test = num_reduced_space_dofs(test)
  nt_test = num_reduced_times(test)
  time_prod_cache = allocate_vector(V,nt_test)
  kron_prod_cache = allocate_vector(V,ns_test*nt_test)
  lincomb_cache = allocate_param_array(kron_prod_cache,num_params(r))
  return time_prod_cache,kron_prod_cache,lincomb_cache
end

function allocate_mdeim_lincomb(
  trial::RBSpace,
  test::RBSpace,
  r::AbstractParamRealization)

  V = get_vector_type(test)
  ns_trial = num_reduced_space_dofs(trial)
  nt_trial = num_reduced_times(trial)
  ns_test = num_reduced_space_dofs(test)
  nt_test = num_reduced_times(test)
  time_prod_cache = allocate_matrix(V,nt_trial,nt_test)
  kron_prod_cache = allocate_matrix(V,ns_trial*nt_trial,ns_test*nt_test)
  lincomb_cache = allocate_param_array(kron_prod_cache,num_params(r))
  return time_prod_cache,kron_prod_cache,lincomb_cache
end

function mdeim_lincomb!(
  cache,
  a::AffineDecomposition{M,A,B,C,<:AbstractMatrix},
  coeff::ParamMatrix) where {M,A,B,C}

  time_prod_cache,kron_prod_cache,lincomb_cache = cache
  basis_time = a.metadata
  basis_space = a.basis_space

  @inbounds for i = eachindex(lincomb_cache)
    lci = lincomb_cache[i]
    ci = coeff[i]
    for j = axes(coeff,2)
      time_prod_cache .= basis_time'*ci[:,j]
      LinearAlgebra.kron!(kron_prod_cache,basis_space[j],time_prod_cache)
      lci .+= kron_prod_cache
    end
  end
end

function mdeim_lincomb!(
  cache,
  a::AffineDecomposition{M,A,B,C,<:AbstractArray},
  coeff::ParamMatrix) where {M,A,B,C}

  time_prod_cache,kron_prod_cache,lincomb_cache = cache
  basis_time = a.metadata
  basis_space = a.basis_space

  @inbounds for i = eachindex(lincomb_cache)
    lci = lincomb_cache[i]
    ci = coeff[i]
    for j = axes(coeff,2)
      for col in axes(basis_time,3)
        for row in axes(basis_time,2)
          time_prod_cache[row,col] = sum(basis_time[:,row,col].*ci[:,j])
        end
      end
      LinearAlgebra.kron!(kron_prod_cache,basis_space[j],time_prod_cache)
      lci .+= kron_prod_cache
    end
  end
end

function mdeim_lincomb!(
  cache,
  a::AffineContribution,
  b::ArrayContribution)

  for (trian,atrian) in a.dict
    mdeim_lincomb!(cache,atrian,b[trian])
  end
end
