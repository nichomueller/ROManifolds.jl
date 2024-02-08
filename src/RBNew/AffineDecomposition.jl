# OFFLINE PHASE

function get_mdeim_indices(A::AbstractMatrix{T}) where T
  m,n = size(A)
  proj = zeros(T,m)
  res = zeros(T,m)
  I = zeros(Int,n)
  I[1] = argmax(abs.(A[:,1]))

  if n > 1
    @inbounds for i = 2:n
      # Bi = view(A,:,1:i-1)
      # Ci = view(A,I[1:i-1],1:i-1)
      # Di = view(A,I[1:i-1],i)
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
  map(eachcol(A)) do a
    basis_test'*a
  end
end

function project_basis_space(A::NnzSnapshots,trial::RBSpace,test::RBSpace)
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
union_indices_time(i::ReducedIntegrationDomain...) = union(map(get_indices_time,i))

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

function Algebra.allocate_in_range(
  a::AffineDecomposition{SpaceOnlyMDEIM},
  r::AbstractParamRealization)

  matrix = get_interp_matrix(a).factors
  v = allocate_in_range(matrix)
  allocate_param_array(v,num_params(r)*num_times(a))
end

function Algebra.allocate_in_range(
  a::AffineDecomposition{SpaceTimeMDEIM},
  r::AbstractParamRealization)

  matrix = get_interp_matrix(a).factors
  v = allocate_in_range(matrix)
  allocate_param_array(v,num_params(r))
end

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
  rbinfo::RBInfo,
  op::RBOperator,
  trian::Triangulation,
  basis_space::AbstractMatrix,
  basis_time::AbstractMatrix)

  indices_space = get_mdeim_indices(basis_space)
  interp_basis_space = view(basis_space,indices_space,:)
  indices_time,lu_interp = _time_indices_and_interp_matrix(rbinfo.mdeim_style,interp_basis_space,basis_time)
  rindices_space = recast_indices_space(basis_space,indices_space)
  red_trian = reduce_triangulation(op,trian,indices_space)
  integration_domain = ReducedIntegrationDomain(rindices_space,indices_time)
  return lu_interp,red_trian,integration_domain
end

const AffineContribution = Contribution{AffineDecomposition}

affine_contribution() = Contribution(IdDict{Triangulation,AffineDecomposition}())

function reduced_vector_form!(
  a::AffineContribution,
  rbinfo::RBInfo,
  op::RBOperator,
  s::AbstractTransientSnapshots,
  trian::Triangulation)

  test = op.test
  basis_space,basis_time = reduced_basis(s;ϵ=get_tol(rbinfo))
  lu_interp,red_trian,integration_domain = mdeim(rbinfo,op,trian,basis_space,basis_time)
  proj_basis_space = project_basis_space(basis_space,test)
  comb_basis_time = combine_basis_time(test)
  a[red_trian] = AffineDecomposition(
    rbinfo.mdeim_style,
    proj_basis_space,
    basis_time,
    lu_interp,
    integration_domain,
    comb_basis_time)
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
  basis_space,basis_time = reduced_basis(s;ϵ=get_tol(rbinfo))
  lu_interp,red_trian,integration_domain = mdeim(rbinfo,op,trian,basis_space,basis_time)
  proj_basis_space = project_basis_space(basis_space,trial,test)
  comb_basis_time = combine_basis_time(trial,test;kwargs...)
  a[red_trian] = AffineDecomposition(
    rbinfo.mdeim_style,
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

  nparams = num_mdeim_params(solver.info)
  smdeim = select_snapshots(s,Base.OneTo(nparams))
  contribs_mat,contribs_vec, = collect_matrices_vectors!(solver,op,smdeim,nothing)
  red_mat = reduced_matrix_form(solver,op,contribs_mat)
  red_vec = reduced_vector_form(solver,op,contribs_vec)
  return red_mat,red_vec
end

# ONLINE PHASE

function allocate_coeff_matrix(a::AffineDecomposition,r::AbstractParamRealization)
  T = eltype(get_interp_matrix(a).factors)
  Nt = num_times(a)
  ns = num_reduced_space_dofs(a)
  m = zeros(T,Nt,ns)
  allocate_param_array(m,num_params(r))
end

function allocate_mdeim_coeff(a::AffineContribution,r::AbstractParamRealization)
  cache_solve = array_contribution()
  cache_recast = array_contribution()
  for (trian,values) in a.dict
    cache_solve[trian] = allocate_in_range(values,r)
    cache_recast[trian] = allocate_coeff_matrix(values,r)
  end
  return cache_solve,cache_recast
end

function allocate_mdeim_coeff(
  mat::Tuple{Vararg{AffineContribution}},
  vec::AffineContribution,
  r::AbstractParamRealization)

  mat_cache = ()
  for mati in mat
    mat_cache = (mat_cache...,allocate_mdeim_coeff(mati,r))
  end
  vec_cache = allocate_mdeim_coeff(vec,r)
  return mat_cache,vec_cache
end

function mdeim_coeff!(
  cache,
  a::AffineDecomposition{SpaceOnlyMDEIM},
  b::ParamVector)

  coeff,coeff_recast = cache
  mdeim_interpolation = a.mdeim_interpolation
  ldiv!(coeff,mdeim_interpolation,b)
  coeff_recast .= transpose(coeff)
end

function mdeim_coeff!(
  cache,
  a::AffineDecomposition{SpaceTimeMDEIM},
  b::ParamVector)

  coeff,coeff_recast = cache
  mdeim_interpolation = a.mdeim_interpolation
  ldiv!(coeff,mdeim_interpolation,b)
  ns = num_reduced_space_dofs(a)
  nt = num_reduced_times(a)
  for j in 1:ns
    sorted_idx = [(i-1)*ns+j for i = 1:nt]
    @inbounds for i = eachindex(new_coeff)
      coeff_recast[i][:,j] = a.basis_time*coeff[i][sorted_idx]
    end
  end
end

function mdeim_coeff!(
  cache,
  a::AffineDecomposition{SpaceTimeMDEIM},
  s::AbstractTransientSnapshots)

  snew = InnerTimeOuterParamTransientSnapshots(s)
  mdeim_coeff!(cache,a,snew.values)
end

function mdeim_coeff!(
  cache,
  a::AffineContribution,
  b::ArrayContribution)

  coeff,coeff_recast = cache
  trians = get_domains(a)
  @check trians == get_domains(coeff) == get_domains(coeff_recast) == get_domains(b)

  for trian in trians
    cache_trian = coeff[trian],coeff_recast[trian]
    a_trian = a[trian]
    b_trian = b[trian]
    mdeim_coeff!(cache_trian,a_trian,b_trian)
  end
end

function mdeim_coeff!(
  cache,
  mat::Tuple{Vararg{AffineContribution}},
  vec::AffineContribution,
  A::Tuple{Vararg{ArrayContribution}},
  b::ArrayContribution)

  cache_mat,cache_vec = cache
  mdeim_coeff!(cache_vec,vec,b)
  map(cache_mat,A,mat) do cache_mat,A,mat
    mdeim_coeff!(cache_mat,A,mat)
  end
  coeff_mat = map(last,cache_mat)
  coeff_vec = last(cache_vec)
  return coeff_mat,coeff_vec
end

function Algebra.allocate_matrix(::Type{V},m::Integer,n::Integer) where V
  T = eltype(V)
  zeros(T,m,n)
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

function allocate_mdeim_lincomb(
  test::RBSpace,
  a::AffineContribution,
  r::AbstractParamRealization)

  time_prod_cache = array_contribution()
  kron_prod_cache = array_contribution()
  lincomb_cache = array_contribution()
  for (trian,values) in a.dict
    ct,ck,cl = allocate_mdeim_lincomb(test,r)
    time_prod_cache[trian] = ct
    kron_prod_cache[trian] = ck
    lincomb_cache[trian] = cl
  end
  return time_prod_cache,kron_prod_cache,lincomb_cache
end

function allocate_mdeim_lincomb(
  trial::RBSpace,
  test::RBSpace,
  a::AffineContribution,
  r::AbstractParamRealization)

  time_prod_cache = array_contribution()
  kron_prod_cache = array_contribution()
  lincomb_cache = array_contribution()
  for (trian,values) in a.dict
    ct,ck,cl = allocate_mdeim_lincomb(trial,test,r)
    time_prod_cache[trian] = ct
    kron_prod_cache[trian] = ck
    lincomb_cache[trian] = cl
  end
  return time_prod_cache,kron_prod_cache,lincomb_cache
end

function allocate_mdeim_lincomb(
  trial::RBSpace,
  test::RBSpace,
  mat::Tuple{Vararg{AffineContribution}},
  vec::AffineContribution,
  r::AbstractParamRealization)

  mat_cache = ()
  for mati in mat
    mat_cache = (mat_cache...,allocate_mdeim_lincomb(trial,test,mati,r))
  end
  vec_cache = allocate_mdeim_lincomb(test,vec,r)
  return mat_cache,vec_cache
end

function mdeim_lincomb!(
  cache,
  a::AffineDecomposition{A,B,C,<:AbstractMatrix},
  coeff::ParamMatrix) where {A,B,C}

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
  a::AffineDecomposition{A,B,C,<:AbstractArray},
  coeff::ParamMatrix) where {A,B,C}

  time_prod_cache,kron_prod_cache,lincomb_cache = cache
  basis_time = a.metadata
  basis_space = a.basis_space

  @inbounds for i = eachindex(lincomb_cache)
    lci = lincomb_cache[i]
    ci = coeff[i]
    for j = axes(coeff,2)
      for col in axes(basis_time,3)
        for row in axes(basis_time,2)
          @fastmath time_prod_cache[row,col] = sum(basis_time[:,row,col].*ci[:,j])
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

  ct,ck,cl = cache
  trians = get_domains(a)
  @check trians == get_domains(ct) == get_domains(ck) == get_domains(cl) == get_domains(b)

  for trian in trians
    cache_trian = ct[trian],ck[trian],cl[trian]
    a_trian = a[trian]
    b_trian = b[trian]
    mdeim_lincomb!(cache_trian,a_trian,b_trian)
  end
end

function mdeim_lincomb!(
  cache,
  mat::Tuple{Vararg{AffineContribution}},
  vec::AffineContribution,
  A::Tuple{Vararg{ArrayContribution}},
  b::ArrayContribution)

  cache_mat,cache_vec = cache
  mdeim_lincomb!(cache_vec,vec,b)
  map(cache_mat,A,mat) do cache_mat,A,mat
    mdeim_lincomb!(cache_mat,A,mat)
  end
  red_mat = map(last,cache_mat)
  red_vec = last(cache_vec)
  return red_mat,red_vec
end
