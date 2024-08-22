# OFFLINE PHASE

function empirical_interpolation!(cache,A::AbstractMatrix)
  I,res = cache
  m,n = size(A)
  resize!(res,m)
  resize!(I,n)
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
  Ai = view(A,I,:)
  return I,Ai
end

function empirical_interpolation!(cache,C::AbstractArray{T,3}) where T
  @check size(C,1) == 1
  c...,Iv = cache
  A = dropdims(C;dims=1)
  I,Ai = empirical_interpolation!(c,A)
  push!(Iv,copy(I))
  return I,Ai
end

function empirical_interpolation!(cache,C::SparseCore)
  @check size(C,1) == 1
  c...,Iv = cache
  A = dropdims(C.array;dims=1)
  I,Ai = empirical_interpolation!(c,A)
  push!(Iv,copy(I))
  return I,Ai
end

"""
    empirical_interpolation(A::AbstractMatrix) -> (Vector{Int}, AbstractMatrix)

Returns a list of indices U+1D4D8 corresponding to the rows of `A` selected by the
discrete empirical interpolation method, and `A[U+1D4D8, :]`

"""
function empirical_interpolation(A::AbstractMatrix)
  m,n = size(A)
  res = zeros(eltype(A),m)
  I = zeros(Int32,n)
  I,Ai = empirical_interpolation!((I,res),A)
  return I,Ai
end

function empirical_interpolation(A::ParamSparseMatrix)
  I,Ai = empirical_interpolation(A.data)
  I′ = recast_indices(I,param_getindex(A,1))
  return I′,Ai
end

function _global_index(i,local_indices::Vector{Vector{Int32}})
  Iprev...,Ig = local_indices
  if length(Iprev) == 0
    return i
  end
  Il = last(Iprev)
  rankl = length(Il)
  islow = slow_index(i,rankl)
  ifast = fast_index(i,rankl)
  iprev = Il[ifast]
  giprev = _global_index(iprev,Iprev)
  return (giprev...,islow)
end

function _global_index(i,Il::Vector{Int32})
  rankl = length(Il)
  li = Il[fast_index(i,rankl)]
  gi = slow_index(i,rankl)
  return li,gi
end

function _to_split_global_indices(local_indices::Vector{Vector{Int32}},index_map::AbstractIndexMap)
  Is...,It = local_indices
  Igt = It
  Igs = copy(It)
  for (i,ii) in enumerate(Igt)
    ilsi,igti = _global_index(ii,last(Is))
    Igt[i] = igti
    Igs[i] = index_map[CartesianIndex(_global_index(ilsi,Is))]
  end
  return Igs,Igt
end

function _to_global_indices(local_indices::Vector{Vector{Int32}},index_map::AbstractIndexMap)
  if length(local_indices) != ndims(index_map) # this is the transient case
    @notimplementedif length(local_indices) != ndims(index_map)+1
    return _to_split_global_indices(local_indices,index_map)
  end
  Ig = local_indices[end]
  for (i,ii) in enumerate(Ig)
    Ig[i] = index_map[CartesianIndex(_global_index(ii,local_indices))]
  end
  return Ig
end

function _eim_cache(C::AbstractArray{T,3}) where T
  m,n = size(C,2),size(C,1)
  res = zeros(T,m)
  I = zeros(Int32,n)
  Iv = Vector{Int32}[]
  return C,I,res,Iv
end

_eim_cache(C::SparseCore) = _eim_cache(C.array)

function _next_core(Aprev::AbstractMatrix{T},Cnext::AbstractArray{T,3}) where T
  Cprev = reshape(Aprev,1,size(Aprev)...)
  _cores2basis(Cprev,Cnext)
end

_next_core(Aprev::AbstractMatrix{T},Cnext::SparseCore{T}) where T = _next_core(Aprev,Cnext.array)

function empirical_interpolation(index_map::AbstractIndexMap,cores::AbstractArray...)
  C,I,res,Iv = _eim_cache(first(cores))
  for i = eachindex(cores)
    _,Ai = empirical_interpolation!((I,res,Iv),C)
    if i < length(cores)
      C = _next_core(Ai,cores[i+1])
    else
      Ig = _to_global_indices(Iv,index_map)
      return Ig,Ai
    end
  end
end

function empirical_interpolation(index_map::SparseIndexMap,cores::AbstractArray...)
  empirical_interpolation(get_sparse_index_map(index_map),cores...)
end

"""
    abstract type AbstractIntegrationDomain end

Type representing the full order dofs selected by an empirical interpolation method.

Subtypes:
- [`IntegrationDomain`](@ref)
- [`TransientIntegrationDomain`](@ref)

"""
abstract type AbstractIntegrationDomain end

get_indices_space(i::AbstractIntegrationDomain) = @abstractmethod
union_indices_space(i::AbstractIntegrationDomain...) = union(map(get_indices_space,i)...)

"""
"""
struct IntegrationDomain{S<:AbstractVector} <: AbstractIntegrationDomain
  indices_space::S
end

get_indices_space(i::IntegrationDomain) = i.indices_space

"""
    get_reduced_cells(cell_dof_ids,dofs::AbstractVector) -> AbstractVector

Returns the list of FE cells containing at least one dof in `dofs`

"""
function get_reduced_cells(cell_dof_ids,dofs::AbstractVector)
  cells = eltype(eltype(cell_dof_ids))[]
  for (cell,celldofs) = enumerate(cell_dof_ids)
    if !isempty(intersect(dofs,celldofs))
      append!(cells,cell)
    end
  end
  return unique(cells)
end

function get_reduced_cells(
  trian::Triangulation,
  ids::AbstractVector,
  test::FESpace)

  cell_dof_ids = get_cell_dof_ids(test,trian)
  indices_space_rows = fast_index(ids,num_free_dofs(test))
  red_integr_cells = get_reduced_cells(cell_dof_ids,indices_space_rows)
  return red_integr_cells
end

function get_reduced_cells(
  trian::Triangulation,
  ids::AbstractVector,
  trial::FESpace,
  test::FESpace)

  cell_dof_ids_trial = get_cell_dof_ids(trial,trian)
  cell_dof_ids_test = get_cell_dof_ids(test,trian)
  indices_space_cols = slow_index(ids,num_free_dofs(test))
  indices_space_rows = fast_index(ids,num_free_dofs(test))
  red_integr_cells_trial = get_reduced_cells(cell_dof_ids_trial,indices_space_cols)
  red_integr_cells_test = get_reduced_cells(cell_dof_ids_test,indices_space_rows)
  red_integr_cells = union(red_integr_cells_trial,red_integr_cells_test)
  return red_integr_cells
end

function reduce_triangulation(trian::Triangulation,i::AbstractIntegrationDomain,r::FESubspace...)
  f = map(get_space,r)
  indices_space = get_indices_space(i)
  red_integr_cells = get_reduced_cells(trian,indices_space,f...)
  red_trian = view(trian,red_integr_cells)
  return red_trian
end

function Algebra.allocate_matrix(::Type{M},m::Integer,n::Integer) where M
  T = eltype(M)
  zeros(T,m,n)
end

function allocate_coefficient(solver::RBSolver,b::Projection)
  n = num_reduced_dofs(b)
  nparams = num_online_params(solver)
  coeffvec = allocate_vector(Vector{Float64},n)
  coeff = array_of_consecutive_arrays(coeffvec,nparams)
  return coeff
end

function allocate_result(solver::RBSolver,test::FESubspace)
  V = get_vector_type(test)
  nfree_test = num_free_dofs(test)
  nparams = num_online_params(solver)
  kronprod = allocate_vector(V,nfree_test)
  result = array_of_consecutive_arrays(kronprod,nparams)
  return result
end

function allocate_result(solver::RBSolver,trial::FESubspace,test::FESubspace)
  T = get_dof_value_type(test)
  nfree_trial = num_free_dofs(trial)
  nfree_test = num_free_dofs(test)
  nparams = num_online_params(solver)
  kronprod = allocate_matrix(Matrix{T},nfree_test,nfree_trial)
  result = array_of_consecutive_arrays(kronprod,nparams)
  return result
end

"""
    struct AffineDecomposition{A,B,C,D,E} end

Stores an affine decomposition of a (discrete) residual/jacobian obtained with
and empirical interpolation method. Its fields are:
- `basis`: the affine terms, it's a subtype of [`Projeciton`](@ref)
- `mdeim_interpolation`: consists of a LU decomposition of the `basis` whose rows
  are restricted to the field `integration_domain`
- `integration_domain`: computed by running the function [`empirical_interpolation`](@ref)
  on the basis, it's a subtype of [`AbstractIntegrationDomain`](@ref)
- `coefficient`: coefficient with respect to the `basis`, cheaply computed thanks
  to the interpolation hypothesis

Note: in order to minimize the memory footprint of the method, the `basis` is
projected on the reduced test/trial subspaces. In other words, it is not properly
a basis for a residual/jacobian, rather it is its (Petrov-) Galerkin projection

"""
struct AffineDecomposition{A,B,C,D,E}
  basis::A
  mdeim_interpolation::B
  integration_domain::C
  coefficient::D
  result::E
end

get_integration_domain(a::AffineDecomposition) = a.integration_domain
get_interp_matrix(a::AffineDecomposition) = a.mdeim_interpolation
get_indices_space(a::AffineDecomposition) = get_indices_space(get_integration_domain(a))

function mdeim(mdeim_style::MDEIMStyle,b::SteadyProjection)
  basis_space = get_basis_space(b)
  indices_space,interp_basis_space = empirical_interpolation(basis_space)
  lu_interp = lu(interp_basis_space)
  integration_domain = IntegrationDomain(indices_space)
  return lu_interp,integration_domain
end

function ParamDataStructures.Contribution(v::Tuple{Vararg{AffineDecomposition}},t::Tuple{Vararg{Triangulation}})
  AffineContribution(v,t)
end

"""
    struct AffineContribution{A,V,K} <: Contribution

The values of an AffineContribution are AffineDecompositions

"""
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

"""
    reduced_form(solver::RBSolver, s::AbstractSnapshots, trian::Triangulation, args...; kwargs...
      ) -> AffineDecomposition, Triangulation

Returns the AffineDecomposition corresponding to the couple (`s`, `trian`)

"""
function reduced_form(
  solver::RBSolver,
  s::AbstractSnapshots,
  trian::Triangulation,
  args...;
  kwargs...)

  mdeim_style = solver.mdeim_style
  basis = reduced_basis(s;ϵ=get_tol(solver))
  lu_interp,integration_domain = mdeim(mdeim_style,basis)
  proj_basis = reduce_operator(mdeim_style,basis,args...;kwargs...)
  red_trian = reduce_triangulation(trian,integration_domain,args...)
  coefficient = allocate_coefficient(solver,basis)
  result = allocate_result(solver,args...)
  ad = AffineDecomposition(proj_basis,lu_interp,integration_domain,coefficient,result)
  return ad,red_trian
end

function reduced_residual(solver::RBSolver,op,s::AbstractSnapshots,trian::Triangulation)
  test = get_test(op)
  reduced_form(solver,s,trian,test)
end

function reduced_jacobian(solver::RBSolver,op,s::AbstractSnapshots,trian::Triangulation;kwargs...)
  trial = get_trial(op)
  test = get_test(op)
  reduced_form(solver,s,trian,trial,test;kwargs...)
end

"""
    reduced_residual(solver::RBSolver,op::PGOperator,c::ArrayContribution)
      ) -> AffineContribution
    reduced_residual(solver::RBSolver,op::TransientPGOperator,c::ArrayContribution)
      ) -> AffineContribution

Returns the AffineContribution corresponding to the residual snapshots stored
in the [`ArrayContribution`](@ref) `c`

"""
function reduced_residual(solver::RBSolver,op,c::ArrayContribution)
  a,trians = map(get_domains(c),get_values(c)) do trian,values
    reduced_residual(solver,op,values,trian)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

"""
    reduced_jacobian(solver::RBSolver,op::PGOperator,c::ArrayContribution);kwargs...
      ) -> AffineContribution
    reduced_jacobian(solver::RBSolver,op::TransientPGOperator,c::TupOfArrayContribution);
      kwargs...) -> AffineContribution

Returns the AffineContribution corresponding to the jacobian snapshots stored
in the [`ArrayContribution`](@ref) `c`. In transient problems, this procedure is
run for every order of the time derivative

"""
function reduced_jacobian(solver::RBSolver,op,c::ArrayContribution;kwargs...)
  a,trians = map(get_domains(c),get_values(c)) do trian,values
    reduced_jacobian(solver,op,values,trian;kwargs...)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

function reduced_jacobian_residual(solver::RBSolver,op,s)
  timer = get_timer(solver)
  jac,res = jacobian_and_residual(solver,op,s)
  @timeit timer "MDEIM" begin
    red_jac = reduced_jacobian(solver,op,jac)
    red_res = reduced_residual(solver,op,res)
  end
  show(timer)
  return red_jac,red_res
end

# ONLINE PHASE

function expand_cache!(a::AffineDecomposition,b::AbstractParamArray)
  coeff = a.coefficient
  result = a.result
  @check param_length(coeff) == param_length(result)
  param_length(coeff) == param_length(b) && return
  a.coefficient.data .= similar(coeff,eltype(coeff),innersize(coeff)...,param_length(b))
  a.result.data .= similar(coeff,eltype(result),innersize(result)...,param_length(b))
end

"""
    coefficient!(a::AffineDecomposition,b::AbstractParamArray) -> AbstractParamArray

Computes the MDEIM coefficient corresponding to the interpolated basis stored in
`a`, with respect to the interpolated snapshots `b`

"""
function coefficient!(a::AffineDecomposition,b::AbstractParamArray)
  coefficient = a.coefficient
  mdeim_interpolation = a.mdeim_interpolation
  ldiv!(coefficient,mdeim_interpolation,b)
end

"""
    mdeim_result(a::AffineDecomposition,b::AbstractParamArray) -> AbstractParamArray

Returns the linear combination of the affine basis by the interpolated coefficient

"""
function mdeim_result(a::AffineDecomposition,b::AbstractParamArray)
  expand_cache!(a,b)
  coefficient!(a,b)

  basis = a.basis
  coefficient = a.coefficient
  result = a.result

  fill!(result,zero(eltype(result)))

  @inbounds for i = eachindex(result)
    result[i] = basis*coefficient[i]
  end

  return result
end

function mdeim_result(a::AffineContribution,b::ArrayContribution)
  @assert length(a) == length(b)
  result = mdeim_result.(a.values,b.values)
  sum(result)
end

# multi field interface

function allocate_result(solver::RBSolver,test::MultiFieldRBSpace)
  active_block_ids = get_touched_blocks(test)
  block_result = [allocate_result(solver,test[i]) for i in active_block_ids]
  return mortar(block_result)
end

function allocate_result(solver::RBSolver,trial::MultiFieldRBSpace,test::MultiFieldRBSpace)
  active_block_ids = Iterators.product(get_touched_blocks(test),get_touched_blocks(trial))
  block_result = [allocate_result(solver,trial[j],test[i]) for (i,j) in active_block_ids]
  return mortar(block_result)
end

struct BlockAffineDecomposition{A,N,C} <: AbstractArray{A,N}
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

function BlockAffineDecomposition(k::BlockMap{N},a::AbstractArray{A},cache) where {A<:AffineDecomposition,N}
  array = Array{A,N}(undef,k.size)
  touched = fill(false,k.size)
  for (t,i) in enumerate(k.indices)
    array[i] = a[t]
    touched[i] = true
  end
  BlockAffineDecomposition(array,touched,cache)
end

Base.size(a::BlockAffineDecomposition,i...) = size(a.array,i...)

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
    error("This block affine decomposition structure is empty")
  end
end

function ParamDataStructures.Contribution(v::Tuple{Vararg{BlockAffineDecomposition}},t::Tuple{Vararg{Triangulation}})
  AffineContribution(v,t)
end

function get_touched_blocks(a::BlockAffineDecomposition)
  findall(a.touched)
end

for f in (:get_integration_domain,:get_interp_matrix,:get_indices_space)
  @eval begin
    function $f(a::BlockAffineDecomposition)
      active_block_ids = get_touched_blocks(a)
      block_map = BlockMap(size(a),active_block_ids)
      blocks = [$f(a[i]) for i in active_block_ids]
      return_cache(block_map,blocks...)
    end
  end
end

function reduce_triangulation(trian::Triangulation,idom::VectorBlock,test::MultiFieldRBSpace)
  active_block_ids = findall(idom.touched)
  red_trian = [reduce_triangulation(trian,idom[i],test[i]) for i in active_block_ids] |> tuple_of_arrays
  return red_trian
end

function reduce_triangulation(trian::Triangulation,idom::MatrixBlock,trial::MultiFieldRBSpace,test::MultiFieldRBSpace)
  active_block_ids = findall(idom.touched)
  red_trian = [reduce_triangulation(trian,idom[i,j],trial[j],test[i]) for (i,j) in Tuple.(active_block_ids)] |> tuple_of_arrays
  return red_trian
end

function reduced_residual(
  solver::RBSolver,
  op,
  s::BlockSnapshots,
  trian::Triangulation)

  test = get_test(op)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  ads,red_trians = [
    reduced_form(solver,s[i],trian,test[i]) for i in active_block_ids
    ] |> tuple_of_arrays
  red_trian = ParamDataStructures.merge_triangulations(red_trians)
  cache = allocate_result(solver,test)
  ad = BlockAffineDecomposition(block_map,ads,cache)
  return ad,red_trian
end

function reduced_jacobian(
  solver::RBSolver,
  op,
  s::BlockSnapshots,
  trian::Triangulation;
  kwargs...)

  trial = get_trial(op)
  test = get_test(op)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  ads,red_trians = [reduced_form(solver,s[i,j],trian,trial[j],test[i];kwargs...)
    for (i,j) in Tuple.(active_block_ids)] |> tuple_of_arrays
  red_trian = ParamDataStructures.merge_triangulations(red_trians)
  cache = allocate_result(solver,trial,test)
  ad = BlockAffineDecomposition(block_map,ads,cache)
  return ad,red_trian
end

function mdeim_result(a::BlockAffineDecomposition,b::ArrayBlock)
  fill!(a.cache,zero(eltype(a.cache)))
  active_block_ids = get_touched_blocks(a)
  for i in Tuple.(active_block_ids)
    a.cache[Block(i)] = mdeim_result(a[i...],b[i...])
  end
  return a.cache
end

# for testing/visualization purposes

struct InterpolationError{A,B}
  name::String
  err_matrix::A
  err_vector::B
  function InterpolationError(err_matrix::A,err_vector::B;name="linear") where {A,B}
    new{A,B}(name,err_matrix,err_vector)
  end
end

function Base.show(io::IO,k::MIME"text/plain",err::InterpolationError)
  print(io,"Interpolation error $(err.name) (matrix,vector): ($(err.err_matrix),$(err.err_vector))")
end

struct LincombError{A,B}
  name::String
  err_matrix::A
  err_vector::B
  function LincombError(err_matrix::A,err_vector::B;name="linear") where {A,B}
    new{A,B}(name,err_matrix,err_vector)
  end
end

function Base.show(io::IO,k::MIME"text/plain",err::LincombError)
  print(io,"Projection error $(err.name) (matrix,vector): ($(err.err_matrix),$(err.err_vector))")
end

function project(A::AbstractMatrix,r::RBSpace)
  basis_space = get_basis_space(r)
  a = basis_space'*A
  v = vec(a)
  return v
end

function project(A::ParamSparseMatrix,trial::RBSpace,test::RBSpace)
  basis_space_test = get_basis_space(test)
  basis_space_trial = get_basis_space(trial)
  s_proj_a = basis_space_test'*param_getindex(A,1)*basis_space_trial
  return s_proj_a
end

function project(fesolver,fes::ArrayContribution,args::RBSpace...;kwargs...)
  sum(map(i->project(fes[i],args...;kwargs...),eachindex(fes)))
end

function project(fesolver,fes::ArrayContribution,test::MultiFieldRBSpace;kwargs...)
  active_block_ids = get_touched_blocks(fes[1])
  block_map = BlockMap(size(fes[1]),active_block_ids)
  rb_blocks = map(active_block_ids) do i
    fesi = contribution(fes.trians) do trian
      val = fes[trian]
      val[i]
    end
    testi = test[i]
    project(fesolver,fesi,testi;kwargs...)
  end
  return_cache(block_map,rb_blocks...)
end

function project(fesolver,fes::ArrayContribution,trial::MultiFieldRBSpace,test::MultiFieldRBSpace;kwargs...)
  active_block_ids = get_touched_blocks(fes[1])
  block_map = BlockMap(size(fes[1]),active_block_ids)
  rb_blocks = map(Tuple.(active_block_ids)) do (i,j)
    fesij = contribution(fes.trians) do trian
      val = fes[trian]
      val[i,j]
    end
    trialj = trial[j]
    testi = test[i]
    project(fesolver,fesij,trialj,testi;kwargs...)
  end
  return_cache(block_map,rb_blocks...)
end

function interpolation_error(a::AffineDecomposition,fes::AbstractSteadySnapshots,rbs::AbstractSteadySnapshots)
  ids_space = get_indices_space(a)
  fes_ids = select_snapshots_entries(fes,ids_space)
  rbs_ids = select_snapshots_entries(rbs,ids_space)
  norm(fes_ids - rbs_ids)
end

function interpolation_error(a::BlockAffineDecomposition,fes::BlockSnapshots,rbs::BlockSnapshots)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  errors = [interpolation_error(a[i],fes[i],rbs[i]) for i = get_touched_blocks(a)]
  return_cache(block_map,errors...)
end

function interpolation_error(a::AffineContribution,fes::ArrayContribution,rbs::ArrayContribution)
  sum([interpolation_error(a[i],fes[i],rbs[i]) for i in eachindex(a)])
end

function interpolation_error(solver,feop,rbop,s;kwargs...)
  odeop = get_algebraic_operator(feop)
  feA,feb = jacobian_and_residual(get_fe_solver(solver),odeop,s)
  rbA,rbb = jacobian_and_residual(solver,rbop.op,s)
  errA = interpolation_error(rbop.lhs,feA,rbA)
  errb = interpolation_error(rbop.rhs,feb,rbb)
  return InterpolationError(errA,errb;kwargs...)
end

function interpolation_error(solver,feop::LinearNonlinearParamFEOperator,rbop,s)
  err_lin = interpolation_error(solver,feop.op_linear,rbop.op_linear,s;name="linear")
  err_nlin = interpolation_error(solver,feop.op_nonlinear,rbop.op_nonlinear,s;name="non linear")
  return err_lin,err_nlin
end

function linear_combination_error(solver,feop,rbop,s;kwargs...)
  odeop = get_algebraic_operator(feop)
  fesolver = get_fe_solver(solver)
  feA,feb = jacobian_and_residual(fesolver,odeop,s)
  feA_comp = project(fesolver,feA,get_trial(rbop),get_test(rbop))
  feb_comp = project(fesolver,feb,get_test(rbop))
  rbA,rbb = jacobian_and_residual(solver,rbop,s)
  errA = rel_norm(feA_comp,rbA)
  errb = rel_norm(feb_comp,rbb)
  return LincombError(errA,errb;kwargs...)
end

function linear_combination_error(solver,feop::LinearNonlinearParamFEOperator,rbop,s)
  err_lin = linear_combination_error(solver,feop.op_linear,rbop.op_linear,s;name="linear")
  err_nlin = linear_combination_error(solver,feop.op_nonlinear,rbop.op_nonlinear,s;name="non linear")
  return err_lin,err_nlin
end

function rel_norm(fe,rb)
  norm(fe - rb) / norm(fe)
end

function rel_norm(fea::ArrayBlock,rba::BlockArrayOfArrays)
  active_block_ids = get_touched_blocks(fea)
  block_map = BlockMap(size(fea),active_block_ids)
  rb_array = get_array(rba)
  norms = [rel_norm(fea[i],rb_array[i]) for i in active_block_ids]
  return_cache(block_map,norms...)
end

function mdeim_error(solver,feop,rbop,s)
  s1 = select_snapshots(s,1)
  intp_err = interpolation_error(solver,feop,rbop,s1)
  proj_err = linear_combination_error(solver,feop,rbop,s1)
  return intp_err,proj_err
end
