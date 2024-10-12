# OFFLINE PHASE

function get_mdeim_indices(A::AbstractMatrix{T}) where T
  m,n = size(A)
  res = zeros(T,m)
  I = zeros(Int,n)
  I[1] = argmax(abs.(A[:,1]))
  if n > 1
    @inbounds for i = 2:n
      Bi = A[:,1:i-1]
      Ci = A[I[1:i-1],1:i-1]
      Di = A[I[1:i-1],i]
      res .= A[:,i] - Bi*(Ci \ Di)
      I[i] = argmax(abs.(res))
    end
  end

  return I
end

struct ReducedIntegrationDomain{S<:AbstractVector,T<:AbstractVector}
  indices_space::S
  indices_time::T
end

get_indices_space(i::ReducedIntegrationDomain) = i.indices_space
get_indices_time(i::ReducedIntegrationDomain) = i.indices_time
union_indices_space(i::ReducedIntegrationDomain...) = union(map(get_indices_space,i)...)
union_indices_time(i::ReducedIntegrationDomain...) = union(map(get_indices_time,i)...)

function get_reduced_cells(
  cell_dof_ids::AbstractVector{<:AbstractVector{T}},
  rows::AbstractVector) where T

  cells = T[]
  for (cell,dofs) = enumerate(cell_dof_ids)
    if !isempty(intersect(rows,dofs))
      append!(cells,cell)
    end
  end
  return unique(cells)
end

function reduce_triangulation(
  trian::Triangulation,
  i::ReducedIntegrationDomain,
  test::RBSpace)

  cell_dof_ids = get_cell_dof_ids(test.space,trian)
  indices_space_rows = fast_index(i.indices_space,num_free_dofs(test.space))
  red_integr_cells = get_reduced_cells(cell_dof_ids,indices_space_rows)
  red_trian = view(trian,red_integr_cells)
  return red_trian
end

function reduce_triangulation(
  trian::Triangulation,
  i::ReducedIntegrationDomain,
  trial::RBSpace,
  test::RBSpace)

  trial0 = trial.space(nothing)
  cell_dof_ids_trial = get_cell_dof_ids(trial0,trian)
  cell_dof_ids_test = get_cell_dof_ids(test.space,trian)
  indices_space_cols = slow_index(i.indices_space,num_free_dofs(trial0))
  indices_space_rows = fast_index(i.indices_space,num_free_dofs(test.space))
  red_integr_cells_trial = get_reduced_cells(cell_dof_ids_trial,indices_space_cols)
  red_integr_cells_test = get_reduced_cells(cell_dof_ids_test,indices_space_rows)
  red_integr_cells = union(red_integr_cells_trial,red_integr_cells_test)
  red_trian = view(trian,red_integr_cells)
  return red_trian
end

function Algebra.allocate_matrix(::Type{V},m::Integer,n::Integer) where V
  T = eltype(V)
  zeros(T,m,n)
end

function allocate_coeff_matrix(
  solver::PODMDEIMSolver{S,SpaceOnlyMDEIM} where S,
  b::PODBasis)

  nspace = num_reduced_space_dofs(b)
  ntime = num_times(b)
  nparams = num_online_params(solver)
  allocate_matrix(Vector{Float64},nspace,ntime*nparams)
end

function allocate_coeff_matrix(
  solver::PODMDEIMSolver{S,SpaceTimeMDEIM} where S,
  b::PODBasis)

  nspace = num_reduced_space_dofs(b)
  ntime = num_reduced_times(b)
  nparams = num_online_params(solver)
  allocate_matrix(Vector{Float64},nspace*ntime,nparams)
end

function allocate_coeff_param_array(solver::PODMDEIMSolver,b::PODBasis)
  nspace = num_reduced_space_dofs(b)
  ntime = num_times(b)
  nparams = num_online_params(solver)
  mat = allocate_matrix(Vector{Float64},ntime,nspace)
  allocate_param_array(mat,nparams)
end

function allocate_coeff_param_array(solver::TTPODMDEIMSolver,b::TTSVDCores)
  nvectors = num_reduced_dofs(b)
  nparams = num_online_params(solver)
  vec = allocate_vector(Vector{Float64},nvectors)
  allocate_param_array(vec,nparams)
end

function allocate_mdeim_coeff(solver::PODMDEIMSolver,b::PODBasis)
  cache_solve = allocate_coeff_matrix(solver,b)
  cache_recast = allocate_coeff_param_array(solver,b)
  return cache_solve,cache_recast
end

function allocate_mdeim_coeff(solver::TTPODMDEIMSolver,b::TTSVDCores)
  cache_recast = allocate_coeff_param_array(solver,b)
  return cache_recast
end

function allocate_mdeim_lincomb(solver::PODMDEIMSolver,test::RBSpace)
  V = get_vector_type(test)
  ns_test = num_reduced_space_dofs(test)
  nt_test = num_reduced_times(test)
  nparams = num_online_params(solver)
  time_prod_cache = allocate_vector(V,nt_test)
  kron_prod = allocate_vector(V,ns_test*nt_test)
  lincomb_cache = allocate_param_array(kron_prod,nparams)
  return time_prod_cache,lincomb_cache
end

function allocate_mdeim_lincomb(solver::TTPODMDEIMSolver,test::RBSpace)
  V = get_vector_type(test)
  nfree_test = num_free_dofs(test)
  nparams = num_online_params(solver)
  kron_prod = allocate_vector(V,nfree_test)
  lincomb_cache = allocate_param_array(kron_prod,nparams)
  return lincomb_cache
end

function allocate_mdeim_lincomb(solver::RBSolver,trial::RBSpace,test::RBSpace)
  V = get_vector_type(test)
  nt_trial = num_reduced_times(trial)
  nt_test = num_reduced_times(test)
  nfree_trial = num_free_dofs(trial)
  nfree_test = num_free_dofs(test)
  nparams = num_online_params(solver)
  time_prod_cache = allocate_matrix(V,nt_test,nt_trial)
  kron_prod = allocate_matrix(V,nfree_test,nfree_trial)
  lincomb_cache = allocate_param_array(kron_prod,nparams)
  return time_prod_cache,lincomb_cache
end

function allocate_mdeim_lincomb(solver::TTPODMDEIMSolver,trial::RBSpace,test::RBSpace)
  V = get_vector_type(test)
  nfree_trial = num_free_dofs(trial)
  nfree_test = num_free_dofs(test)
  nparams = num_online_params(solver)
  kron_prod = allocate_matrix(V,nfree_test,nfree_trial)
  lincomb_cache = allocate_param_array(kron_prod,nparams)
  return lincomb_cache
end

function allocate_mdeim_cache(solver::RBSolver,b::Projection,args...)
  coeff_cache = allocate_mdeim_coeff(solver,b)
  lincomb_cache = allocate_mdeim_lincomb(solver,args...)
  return coeff_cache,lincomb_cache
end

struct AffineDecomposition{M,A,B,C,D} <: Projection
  mdeim_style::M
  basis::A
  mdeim_interpolation::B
  integration_domain::C
  cache::D
end

const TTAffineDecomposition = AffineDecomposition{SpaceTimeMDEIM,A,B,C,D} where {
  A<:TTProjection,B,C,D}

get_integration_domain(a::AffineDecomposition) = a.integration_domain
get_interp_matrix(a::AffineDecomposition) = a.mdeim_interpolation
get_indices_space(a::AffineDecomposition) = get_indices_space(get_integration_domain(a))
get_indices_time(a::AffineDecomposition) = get_indices_time(get_integration_domain(a))
num_space_dofs(a::AffineDecomposition) = @notimplemented
FEM.num_times(a::AffineDecomposition) = num_times(a.basis)
num_reduced_space_dofs(a::AffineDecomposition) = length(get_indices_space(a))
num_reduced_times(a::AffineDecomposition) = length(get_indices_time(a))
get_mdeim_cache(a::AffineDecomposition) = a.cache
get_coeff_cache(a::AffineDecomposition) = first(get_mdeim_cache(a))
get_lincomb_cache(a::AffineDecomposition) = last(get_mdeim_cache(a))

function _time_indices_and_interp_matrix(::SpaceTimeMDEIM,interp_basis_space,basis_time)
  indices_time = get_mdeim_indices(basis_time)
  interp_basis_time = view(basis_time,indices_time,:)
  interp_basis_space_time = kronecker(interp_basis_time,interp_basis_space)
  lu_interp = lu(interp_basis_space_time)
  return indices_time,lu_interp
end

function _time_indices_and_interp_matrix(::SpaceOnlyMDEIM,interp_basis_space,basis_time)
  indices_time = axes(basis_time,1)
  lu_interp = lu(interp_basis_space)
  return indices_time,lu_interp
end

function mdeim(solver::RBSolver,b::PODBasis)
  indices_space = get_mdeim_indices(b.basis_space)
  interp_basis_space = view(b.basis_space,indices_space,:)
  indices_time,lu_interp = _time_indices_and_interp_matrix(solver.mdeim_style,interp_basis_space,b.basis_time)
  recast_indices_space = recast_indices(b.basis_space,indices_space)
  integration_domain = ReducedIntegrationDomain(recast_indices_space,indices_time)
  return lu_interp,integration_domain
end

function mdeim(solver::RBSolver,b::TTSVDCores)
  indices_spacetime = get_mdeim_indices(b.basis_spacetime)
  indices_space = fast_index(indices_spacetime,num_space_dofs(b))
  indices_time = slow_index(indices_spacetime,num_space_dofs(b))
  lu_interp = lu(view(b.basis_spacetime,indices_spacetime,:))
  recast_indices_space = recast_indices(b.basis_spacetime,indices_space)
  integration_domain = ReducedIntegrationDomain(recast_indices_space,indices_time)
  return lu_interp,integration_domain
end

function FEM.Contribution(v::Tuple{Vararg{AffineDecomposition}},t::Tuple{Vararg{Triangulation}})
  AffineContribution(v,t)
end

struct AffineContribution{A,V,K} <: Contribution
  values::V
  trians::K
  function AffineContribution(
    values::V,
    trians::K
    ) where {A,V<:Tuple{Vararg{A}},K<:Tuple{Vararg{Triangulation}}}

    @check length(values) == length(trians)
    @check !any([t === first(trians) for t = trians[2:end]])
    new{A,V,K}(values,trians)
  end
end

function union_reduced_times(a::AffineContribution)
  idom = ()
  for values in get_values(a)
    idom = (idom...,get_integration_domain(values))
  end
  union_indices_time(idom...)
end

function union_reduced_times(a::NTuple)
  union([union_reduced_times(ai) for ai in a]...)
end

function reduced_form(
  solver::RBSolver,
  s::S,
  trian::T,
  args...;
  kwargs...) where {S,T}

  basis = reduced_basis(s;ϵ=get_tol(solver))
  lu_interp,integration_domain = mdeim(solver,basis)
  proj_basis = compress_basis(basis,args...;kwargs...)
  red_trian = reduce_triangulation(trian,integration_domain,args...)
  cache = allocate_mdeim_cache(solver,basis,args...)
  ad = AffineDecomposition(solver.mdeim_style,proj_basis,lu_interp,integration_domain,cache)
  return ad,red_trian
end

function reduced_vector_form(
  solver::RBSolver,
  op::RBOperator,
  s::S,
  trian::T) where {S,T}

  test = get_test(op)
  reduced_form(solver,s,trian,test)
end

function reduced_matrix_form(
  solver::RBSolver,
  op::RBOperator,
  s::S,
  trian::T;
  kwargs...) where {S,T}

  trial = get_trial(op)
  test = get_test(op)
  reduced_form(solver,s,trian,trial,test;kwargs...)
end

function reduced_vector_form(
  solver::RBSolver,
  op::RBOperator,
  c::ArrayContribution)

  a,trians = map(get_domains(c),get_values(c)) do trian,values
    reduced_vector_form(solver,op,values,trian)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

function reduced_matrix_form(
  solver::RBSolver,
  op::RBOperator,
  c::ArrayContribution;
  kwargs...)

  a,trians = map(get_domains(c),get_values(c)) do trian,values
    reduced_matrix_form(solver,op,values,trian;kwargs...)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

function reduced_matrix_form(
  solver::ThetaMethodRBSolver,
  op::RBOperator,
  contribs::Tuple{Vararg{Any}})

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
  s::S) where S

  smdeim = select_snapshots(s,mdeim_params(solver))
  contribs_mat,contribs_vec = jacobian_and_residual(solver,op,smdeim)
  red_mat = reduced_matrix_form(solver,op,contribs_mat)
  red_vec = reduced_vector_form(solver,op,contribs_vec)
  return red_mat,red_vec
end

# ONLINE PHASE

function mdeim_coeff(a::AffineDecomposition{SpaceOnlyMDEIM},b::AbstractMatrix)
  coeff,coeff_recast = get_coeff_cache(a)
  mdeim_interpolation = a.mdeim_interpolation
  ldiv!(coeff,mdeim_interpolation,b)
  nt = num_times(a)
  @inbounds for i = eachindex(coeff_recast)
    coeff_recast[i] = transpose(coeff[:,(i-1)*nt+1:i*nt])
  end
  return coeff_recast
end

function mdeim_coeff(a::AffineDecomposition{SpaceTimeMDEIM},b::AbstractMatrix)
  coeff,coeff_recast = get_coeff_cache(a)
  mdeim_interpolation = a.mdeim_interpolation
  basis_time = a.basis.basis_time

  ns = num_reduced_space_dofs(a)
  nt = num_reduced_times(a)
  np = length(coeff_recast)

  bvec = reshape(b,:,np)
  ldiv!(coeff,mdeim_interpolation,bvec)
  @inbounds for i = eachindex(coeff_recast)
    for j in 1:ns
      coeff_recast[i][:,j] = basis_time*coeff[j:ns:ns*nt,i]
    end
  end

  return coeff_recast
end

function mdeim_coeff(a::TTAffineDecomposition,b::ParamVector)
  coeff = get_coeff_cache(a)
  mdeim_interpolation = a.mdeim_interpolation
  @inbounds for i = eachindex(coeff)
    ldiv!(coeff[i],mdeim_interpolation,b[i])
  end
  return coeff
end

function residual_mdeim_lincomb(a::AffineDecomposition,coeff::ParamMatrix)
  time_prod_cache,lincomb_cache = get_lincomb_cache(a)
  fill!(lincomb_cache,zero(eltype(lincomb_cache)))

  basis_time = a.basis.metadata
  basis_space = a.basis.basis_space

  @inbounds for i = eachindex(lincomb_cache)
    lci = lincomb_cache[i]
    ci = coeff[i]
    for j = axes(coeff,2)
      time_prod_cache .= basis_time'*ci[:,j]
      lci .+= kronecker(time_prod_cache,basis_space[j])
    end
  end

  return lincomb_cache
end

function jacobian_mdeim_lincomb(a::AffineDecomposition,coeff::ParamMatrix)
  time_prod_cache,lincomb_cache = get_lincomb_cache(a)
  fill!(lincomb_cache,zero(eltype(lincomb_cache)))

  basis_time = a.basis.metadata
  basis_space = a.basis.basis_space

  @inbounds for i = eachindex(lincomb_cache)
    lci = lincomb_cache[i]
    ci = coeff[i]
    for j = axes(coeff,2)
      for col in axes(basis_time,3)
        for row in axes(basis_time,2)
          time_prod_cache[row,col] = sum(basis_time[:,row,col].*ci[:,j])
        end
      end
      lci .+= kronecker(time_prod_cache,basis_space[j])
    end
  end

  return lincomb_cache
end

function residual_mdeim_lincomb(a::TTAffineDecomposition,coeff::ParamVector)
  lincomb_cache = get_lincomb_cache(a)
  fill!(lincomb_cache,zero(eltype(lincomb_cache)))
  basis_spacetime = a.basis.metadata
  for i = eachindex(lincomb_cache)
    lci = lincomb_cache[i]
    ci = coeff[i]
    for j in eachindex(ci)
      @inbounds lci .+= basis_spacetime[j]*ci[j]
    end
  end
  return lincomb_cache
end

function jacobian_mdeim_lincomb(a::TTAffineDecomposition,coeff::ParamVector)
  lincomb_cache = get_lincomb_cache(a)
  fill!(lincomb_cache,zero(eltype(lincomb_cache)))
  basis_spacetime = a.basis.metadata
  for i = eachindex(lincomb_cache)
    lci = lincomb_cache[i]
    ci = coeff[i]
    for j in eachindex(ci)
      @inbounds lci .+= basis_spacetime[j]*ci[j]
    end
  end
  return lincomb_cache
end

function mdeim_residual(a::AffineContribution,b::ArrayContribution)
  @assert length(a) == length(b)
  ress = map(a.values,b.values) do a,b
    coeff = mdeim_coeff(a,b)
    residual_mdeim_lincomb(a,coeff)
  end
  sum(ress)
end

function mdeim_jacobian(a::AffineContribution,b::ArrayContribution)
  @assert length(a) == length(b)
  jacs = map(a.values,b.values) do a,b
    coeff = mdeim_coeff(a,b)
    jacobian_mdeim_lincomb(a,coeff)
  end
  sum(jacs)
end

function mdeim_jacobian(a::Tuple,b::Tuple)
  sum(map(mdeim_jacobian,a,b))
end

# multi field interface

function allocate_block_mdeim_lincomb(solver::RBSolver,test::RBSpace)
  active_block_ids = get_touched_blocks(test)
  block_lincomb = Any[allocate_mdeim_lincomb(solver,test[i]) for i in active_block_ids]
  _,block_lc = tuple_of_arrays(block_lincomb)
  return mortar(block_lc)
end

function allocate_block_mdeim_lincomb(solver::TTPODMDEIMSolver,test::RBSpace)
  active_block_ids = get_touched_blocks(test)
  block_lincomb = Any[allocate_mdeim_lincomb(solver,test[i]) for i in active_block_ids]
  return mortar(block_lincomb)
end

function allocate_block_mdeim_lincomb(solver::RBSolver,trial::RBSpace,test::RBSpace)
  active_block_ids = Iterators.product(get_touched_blocks(test),get_touched_blocks(trial))
  block_lincomb = Any[allocate_mdeim_lincomb(solver,trial[j],test[i]) for (i,j) in active_block_ids]
  _,block_lc = tuple_of_arrays(block_lincomb)
  return mortar(block_lc)
end

function allocate_block_mdeim_lincomb(solver::TTPODMDEIMSolver,trial::RBSpace,test::RBSpace)
  active_block_ids = Iterators.product(get_touched_blocks(test),get_touched_blocks(trial))
  block_lincomb = Any[allocate_mdeim_lincomb(solver,trial[j],test[i]) for (i,j) in active_block_ids]
  return mortar(block_lincomb)
end

struct BlockAffineDecomposition{A,N,C}
  array::Array{A,N}
  touched::Array{Bool,N}
  cache::C
  function BlockAffineDecomposition(
    array::Array{A,N},
    touched::Array{Bool,N},
    cache::C
    ) where {A<:AffineDecomposition,N,C}

    @check size(array) == size(touched)
    new{A,N,C}(array,touched,cache)
  end
end

function BlockAffineDecomposition(k::BlockMap{N},cache,a::A...) where {A<:AffineDecomposition,N}
  array = Array{A,N}(undef,k.size)
  touched = fill(false,k.size)
  for (t,i) in enumerate(k.indices)
    array[i] = a[t]
    touched[i] = true
  end
  BlockAffineDecomposition(array,touched,cache)
end

Base.size(a::BlockAffineDecomposition,i...) = size(a.array,i...)
Base.length(a::BlockAffineDecomposition) = length(a.array)
Base.eltype(::Type{<:BlockAffineDecomposition{A}}) where A = A
Base.eltype(a::BlockAffineDecomposition{A}) where A = A
Base.ndims(a::BlockAffineDecomposition{A,N}) where {A,N} = N
Base.ndims(::Type{BlockAffineDecomposition{A,N}}) where {A,N} = N
Base.eachindex(a::BlockAffineDecomposition) = eachindex(a.array)
function Base.getindex(a::BlockAffineDecomposition,i...)
  if !a.touched[i...]
    return nothing
  end
  a.array[i...]
end

function Base.setindex!(a::BlockAffineDecomposition,v,i...)
  @check a.touched[i...] "Only touched entries can be set"
  a.array[i...] = v
end

function Arrays.testitem(a::BlockAffineDecomposition)
  i = findall(a.touched)
  if length(i) != 0
    a.array[i[1]]
  else
    error("This block snapshots structure is empty")
  end
end

function FEM.Contribution(v::Tuple{Vararg{BlockAffineDecomposition}},t::Tuple{Vararg{Triangulation}})
  AffineContribution(v,t)
end

function get_touched_blocks(a::BlockAffineDecomposition)
  findall(a.touched)
end

function get_integration_domain(a::BlockAffineDecomposition)
  active_block_ids = get_touched_blocks(a)
  block_indices_space = get_indices_space(a)
  union_indices_space = union([block_indices_space[i] for i in active_block_ids]...)
  union_indices_time = get_indices_time(a)
  ReducedIntegrationDomain(union_indices_space,union_indices_time)
end

function get_interp_matrix(a::BlockAffineDecomposition)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  mdeim_interp = Any[get_interp_matrix(a[i]) for i = active_block_ids(a)]
  return_cache(block_map,mdeim_interp...)
end

function get_indices_space(a::BlockAffineDecomposition)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  indices_space = Any[get_indices_space(a[i]) for i = get_touched_blocks(a)]
  return_cache(block_map,indices_space...)
end

function get_indices_time(a::BlockAffineDecomposition)
  indices_time = Any[get_indices_time(a[i]) for i = get_touched_blocks(a)]
  union(indices_time...)
end

num_space_dofs(a::BlockAffineDecomposition) = @notimplemented

function FEM.num_times(a::BlockAffineDecomposition)
  num_times(testitem(a))
end

function num_reduced_space_dofs(a::BlockAffineDecomposition)
  length(get_indices_space(a))
end

function num_reduced_times(a::BlockAffineDecomposition)
  length(get_indices_time(a))
end

function reduced_vector_form(
  solver::RBSolver,
  op::RBOperator,
  s::BlockSnapshots,
  trian::T) where T

  test = get_test(op)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  ads,red_trians = Any[
    reduced_form(solver,s[i],trian,test[i]) for i in active_block_ids
    ] |> tuple_of_arrays
  red_trian = FEM.merge_triangulations(red_trians)
  cache = allocate_block_mdeim_lincomb(solver,test)
  ad = BlockAffineDecomposition(block_map,cache,ads...)
  return ad,red_trian
end

function reduced_matrix_form(
  solver::RBSolver,
  op::RBOperator,
  s::BlockSnapshots,
  trian::T;
  kwargs...) where T

  trial = get_trial(op)
  test = get_test(op)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  ads,red_trians = Any[
    reduced_form(solver,s[i,j],trian,trial[j],test[i];kwargs...) for (i,j) in Tuple.(active_block_ids)
    ] |> tuple_of_arrays
  red_trian = FEM.merge_triangulations(red_trians)
  cache = allocate_block_mdeim_lincomb(solver,trial,test)
  ad = BlockAffineDecomposition(block_map,cache,ads...)
  return ad,red_trian
end

function mdeim_coeff(a::BlockAffineDecomposition,b::BlockSnapshots)
  @check get_touched_blocks(a) == get_touched_blocks(b)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  coeff = Any[mdeim_coeff(a[i],b[i]) for i in active_block_ids]
  return_cache(block_map,coeff...)
end

function residual_mdeim_lincomb(a::BlockAffineDecomposition,coeff::ArrayBlock)
  active_block_ids = get_touched_blocks(a)
  for i in active_block_ids
    a.cache[Block(i)] = residual_mdeim_lincomb(a[i],coeff[i])
  end
  return a.cache
end

function jacobian_mdeim_lincomb(a::BlockAffineDecomposition,coeff::ArrayBlock)
  active_block_ids = get_touched_blocks(a)
  for i in active_block_ids
    a.cache[Block(i.I)] = jacobian_mdeim_lincomb(a[i],coeff[i])
  end
  return a.cache
end

# for testing/visualization purposes

function compress(A::AbstractMatrix,r::RBSpace)
  basis_space = get_basis_space(r)
  basis_time = get_basis_time(r)

  a = (basis_space'*A)*basis_time
  v = vec(a)
  return v
end

function compress(A::AbstractMatrix{T},trial::RBSpace,test::RBSpace;combine=(x,y)->x) where T
  basis_space_test = get_basis_space(test)
  basis_time_test = get_basis_time(test)
  basis_space_trial = get_basis_space(trial)
  basis_time_trial = get_basis_time(trial)
  ns_test,ns_trial = size(basis_space_test,2),size(basis_space_trial,2)
  nt_test,nt_trial = size(basis_time_test,2),size(basis_time_trial,2)

  red_xvec = compress_basis_space(A,get_basis_space(trial),get_basis_space(test))
  a = stack(vec.(red_xvec))'  # Nt x ns_test*ns_trial
  st_proj = zeros(T,nt_test,nt_trial,ns_test*ns_trial)
  st_proj_shift = zeros(T,nt_test,nt_trial,ns_test*ns_trial)
  @inbounds for ins = 1:ns_test*ns_trial, jt = 1:nt_trial, it = 1:nt_test
    st_proj[it,jt,ins] = sum(basis_time_test[:,it].*basis_time_trial[:,jt].*a[:,ins])
    st_proj_shift[it,jt,ins] = sum(basis_time_test[2:end,it].*basis_time_trial[1:end-1,jt].*a[2:end,ins])
  end
  st_proj = combine(st_proj,st_proj_shift)
  st_proj_a = zeros(T,ns_test*nt_test,ns_trial*nt_trial)
  @inbounds for i = 1:ns_trial, j = 1:ns_test
    st_proj_a[j:ns_test:ns_test*nt_test,i:ns_trial:ns_trial*nt_trial] = st_proj[:,:,(i-1)*ns_test+j]
  end
  return st_proj_a
end

function compress(solver,fes::ArrayContribution,args::RBSpace...;kwargs...)
  sum(map(i->compress(fes[i],args...;kwargs...),eachindex(fes)))
end

function compress(solver,fes::ArrayContribution,test::BlockRBSpace;kwargs...)
  active_block_ids = get_touched_blocks(fes[1])
  block_map = BlockMap(size(fes[1]),active_block_ids)
  rb_blocks = map(active_block_ids) do i
    fesi = contribution(fes.trians) do trian
      val = fes[trian]
      val[i]
    end
    testi = test[i]
    compress(solver,fesi,testi;kwargs...)
  end
  return_cache(block_map,rb_blocks...)
end

function compress(solver,fes::ArrayContribution,trial::BlockRBSpace,test::BlockRBSpace;kwargs...)
  active_block_ids = get_touched_blocks(fes[1])
  block_map = BlockMap(size(fes[1]),active_block_ids)
  rb_blocks = map(Tuple.(active_block_ids)) do (i,j)
    fesij = contribution(fes.trians) do trian
      val = fes[trian]
      val[i,j]
    end
    trialj = trial[j]
    testi = test[i]
    compress(solver,fesij,trialj,testi;kwargs...)
  end
  return_cache(block_map,rb_blocks...)
end

function compress(solver,fes::Tuple{Vararg{ArrayContribution}},trial,test)
  fesolver = get_fe_solver(solver)
  cmp = ()
  for i = eachindex(fes)
    combine = (x,y) -> i == 1 ? fesolver.θ*x+(1-fesolver.θ)*y : fesolver.θ*(x-y)
    cmp = (cmp...,compress(solver,fes[i],trial,test;combine))
  end
  sum(cmp)
end

function interpolation_error(a::AffineDecomposition,fes::AbstractSnapshots,rbs::AbstractSnapshots)
  ids_space,ids_time = get_indices_space(a),get_indices_time(a)
  fes_ids = select_snapshots_entries(reverse_snapshots(fes),ids_space,ids_time)
  rbs_ids = select_snapshots_entries(reverse_snapshots(rbs),ids_space,ids_time)
  norm(fes_ids - rbs_ids)
end

function interpolation_error(a::BlockAffineDecomposition,fes::BlockSnapshots,rbs::BlockSnapshots)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  errors = Any[interpolation_error(a[i],fes[i],rbs[i]) for i = get_touched_blocks(a)]
  return_cache(block_map,errors...)
end

function interpolation_error(a::AffineContribution,fes::ArrayContribution,rbs::ArrayContribution)
  sum([interpolation_error(a[i],fes[i],rbs[i]) for i in eachindex(a)])
end

function interpolation_error(a::Tuple,fes::Tuple,rbs::Tuple)
  @check length(a) == length(fes) == length(rbs)
  err = ()
  for i = eachindex(a)
    err = (err...,interpolation_error(a[i],fes[i],rbs[i]))
  end
  err
end

function _interpolation_error(solver,feop,rbop,s)
  feA,feb = _jacobian_and_residual(get_fe_solver(solver),feop,s)
  rbA,rbb = _jacobian_and_residual(solver,rbop.op,s)
  errA = interpolation_error(rbop.lhs,feA,rbA)
  errb = interpolation_error(rbop.rhs,feb,rbb)
  return errA,errb
end

function interpolation_error(solver,feop,rbop,s)
  errA,errb = _interpolation_error(solver,feop,rbop,s)
  return Dict("interpolation error matrix" => errA),Dict("interpolation error vector" => errb)
end

function interpolation_error(solver,feop::TransientParamLinearNonlinearFEOperator,rbop,s)
  errA_lin,errb_lin = _interpolation_error(solver,feop.op_linear,rbop.op_linear,s)
  errA_nlin,errb_nlin = _interpolation_error(solver,feop.op_nonlinear,rbop.op_nonlinear,s)
  err_lin = (Dict("interpolation error linear matrix" => errA_lin),
    Dict("interpolation error linear vector" => errb_lin))
  err_nlin = (Dict("interpolation error nonlinear matrix" => errA_nlin),
    Dict("interpolation error nonlinear vector" => errb_nlin))
  return err_lin,err_nlin
end

function _linear_combination_error(solver,feop,rbop,s)
  feA,feb = _jacobian_and_residual(get_fe_solver(solver),feop,s)
  feA_comp = compress(solver,feA,get_trial(rbop),get_test(rbop))
  feb_comp = compress(solver,feb,get_test(rbop))
  rbA,rbb = _jacobian_and_residual(solver,rbop,s)
  errA = _rel_norm(feA_comp,rbA)
  errb = _rel_norm(feb_comp,rbb)
  return errA,errb
end

function linear_combination_error(solver,feop,rbop,s)
  errA,errb = _linear_combination_error(solver,feop,rbop,s)
  Dict("projection error matrix" => errA),Dict("projection error vector" => errb)
end

function linear_combination_error(solver,feop::TransientParamLinearNonlinearFEOperator,rbop,s)
  errA_lin,errb_lin = _linear_combination_error(solver,feop.op_linear,rbop.op_linear,s)
  errA_nlin,errb_nlin = _linear_combination_error(solver,feop.op_nonlinear,rbop.op_nonlinear,s)
  err_lin = (Dict("projection error linear matrix" => errA_lin),
    Dict("projection error linear vector" => errb_lin))
  err_nlin = (Dict("projection error nonlinear matrix" => errA_nlin),
    Dict("projection error nonlinear vector" => errb_nlin))
  return err_lin,err_nlin
end

function _rel_norm(fea,rba)
  norm(fea - rba) / norm(fea)
end

function _rel_norm(fea::ArrayBlock,rba::ParamBlockArray)
  active_block_ids = get_touched_blocks(fea)
  block_map = BlockMap(size(fea),active_block_ids)
  rb_array = get_array(rba)
  norms = [_rel_norm(fea[i],rb_array[i]) for i in active_block_ids]
  return_cache(block_map,norms...)
end

function _jacobian_and_residual(solver::ThetaMethodRBSolver,op::RBOperator{C},s) where C
  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ
  r = get_realization(s)
  FEM.shift_time!(r,dt*(θ-1))
  ode_cache = allocate_cache(op,r)
  u0 = get_values(s)
  if C == Affine
    u0 .= 0.0
  end
  vθ = similar(u0)
  vθ .= 0.0
  ode_cache = update_cache!(ode_cache,op,r)
  nlop = RBThetaMethodParamOperator(op,r,dtθ,u0,ode_cache,vθ)
  A = allocate_jacobian(nlop,u0)
  b = allocate_residual(nlop,u0)
  sA = jacobian!(A,nlop,u0)
  sb = residual!(b,nlop,u0)
  sA,sb
end

function _jacobian_and_residual(solver::ThetaMethod,op::ODEParamOperator{C},s) where C
  dt = solver.dt
  θ = solver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ
  r = get_realization(s)
  FEM.shift_time!(r,dt*(θ-1))
  ode_cache = allocate_cache(op,r)
  u0 = get_values(s)
  if C == Affine
    u0 .= 0.0
  end
  vθ = similar(u0)
  vθ .= 0.0
  ode_cache = update_cache!(ode_cache,op,r)
  nlop = ThetaMethodParamOperator(op,r,dtθ,u0,ode_cache,vθ)
  A = allocate_jacobian(nlop,u0)
  b = allocate_residual(nlop,u0)
  jacobian!(A,nlop,u0)
  residual!(b,nlop,u0)
  sA = map(A->Snapshots(A,r),A)
  sb = Snapshots(b,r)
  sA,sb
end

function _jacobian_and_residual(
  solver::ThetaMethod,
  _feop::TransientParamFEOperator,
  args...)

  op = get_algebraic_operator(_feop)
  _jacobian_and_residual(solver,op,args...)
end

function mdeim_error(solver,feop,rbop,s)
  s1 = select_snapshots(s,1)
  intp_err = interpolation_error(solver,feop,rbop,s1)
  proj_err = linear_combination_error(solver,feop,rbop,s1)
  return intp_err,proj_err
end
