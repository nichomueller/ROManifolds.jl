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

function empirical_interpolation!(cache,core::AbstractArray{T,3}) where T
  @check size(C,1) == 1
  _cache...,Iv = cache
  A = dropdims(core;dims=1)
  I,Ai = empirical_interpolation!(_cache,A)
  push!(Iv,copy(I))
  return I,Ai
end

function eim_cache(A::AbstractMatrix)
  m,n = size(A)
  res = zeros(eltype(A),m)
  I = zeros(Int32,n)
  return I,res
end

function eim_cache(cores::Vector{<:AbstractArray{T,3}}) where T
  c = first(cores)
  m,n = size(c,2),size(c,1)
  res = zeros(T,m)
  I = zeros(Int32,n)
  Iv = Vector{Int32}[]
  return I,res,Iv
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

function allocate_coefficient(red::AbstractReduction,b::Projection)
  n = num_reduced_dofs(b)
  coeffvec = allocate_vector(Vector{Float64},n)
  coeff = array_of_consecutive_arrays(coeffvec,num_online_params(red))
  return coeff
end

function allocate_result(red::AbstractReduction,test::FESubspace)
  V = get_vector_type(test)
  nfree_test = num_free_dofs(test)
  b = allocate_vector(V,nfree_test)
  result = array_of_consecutive_arrays(b,num_online_params(red))
  return result
end

function allocate_result(red::AbstractReduction,trial::FESubspace,test::FESubspace)
  T = get_dof_value_type(test)
  nfree_trial = num_free_dofs(trial)
  nfree_test = num_free_dofs(test)
  A = allocate_matrix(Matrix{T},nfree_test,nfree_trial)
  result = array_of_consecutive_arrays(A,num_online_params(red))
  return result
end

"""
    struct AffineDecomposition{A,B,C,D,E} end

Stores an affine decomposition of a (discrete) residual/jacobian obtained with
and empirical interpolation method. Its fields are:
- `basis`: the affine terms, it's a subtype of [`Projeciton`](@ref)
- `interpolation`: consists of a LU decomposition of the `basis` whose rows
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
  interpolation::B
  integration_domain::C
  coefficient::D
  result::E
end

get_integration_domain(a::AffineDecomposition) = a.integration_domain
get_interp_matrix(a::AffineDecomposition) = a.interpolation
get_indices_space(a::AffineDecomposition) = get_indices_space(get_integration_domain(a))

function mdeim(b::Projection)
  basis_space = get_basis_space(b)
  indices_space,interp_basis_space = empirical_interpolation(basis_space)
  interpolation = lu(interp_basis_space)
  integration_domain = IntegrationDomain(indices_space)
  return interpolation,integration_domain
end

function mdeim(b::TTSVDCores)
  index_map = get_index_map(b)
  cores_space = get_cores(b)
  indices_space,interp_basis_space = empirical_interpolation(index_map,cores_space...)
  interpolation = lu(interp_basis_space)
  integration_domain = IntegrationDomain(indices_space)
  return interpolation,integration_domain
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
    reduced_form(red::AbstractReduction, s::AbstractSnapshots, trian::Triangulation, args...; kwargs...
      ) -> AffineDecomposition, Triangulation

Returns the AffineDecomposition corresponding to the couple (`s`, `trian`)

"""
function reduced_form(
  red::AbstractMDEIMReduction,
  s::AbstractSnapshots,
  trian::Triangulation,
  args...)

  t = @timed begin
    basis = reduced_basis(get_reduction(red),s)
    interpolation,integration_domain = mdeim(basis)
    proj_basis = reduce_operator(red,basis,args...)
    red_trian = reduce_triangulation(trian,integration_domain,args...)
  end

  coefficient = allocate_coefficient(red,basis)
  result = allocate_result(red,args...)

  println(CostTracker(t))

  ad = AffineDecomposition(proj_basis,interpolation,integration_domain,coefficient,result)
  return ad,red_trian
end

function reduced_residual(red::AbstractReduction,op,s::AbstractSnapshots,trian::Triangulation)
  test = get_test(op)
  reduced_form(red,s,trian,test)
end

function reduced_jacobian(red::AbstractReduction,op,s::AbstractSnapshots,trian::Triangulation)
  trial = get_trial(op)
  test = get_test(op)
  reduced_form(red,s,trian,trial,test)
end

"""
    reduced_residual(red::AbstractReduction,op::PGOperator,c::ArrayContribution)
      ) -> AffineContribution
    reduced_residual(red::AbstractReduction,op::TransientPGOperator,c::ArrayContribution)
      ) -> AffineContribution

Returns the AffineContribution corresponding to the residual snapshots stored
in the [`ArrayContribution`](@ref) `c`

"""
function reduced_residual(red::AbstractReduction,op,c::ArrayContribution)
  a,trians = map(get_domains(c),get_values(c)) do trian,values
    reduced_residual(red,op,values,trian)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

"""
    reduced_jacobian(red::AbstractReduction,op::PGOperator,c::ArrayContribution)
      ) -> AffineContribution
    reduced_jacobian(red::AbstractReduction,op::TransientPGOperator,c::TupOfArrayContribution)
      ) -> AffineContribution

Returns the AffineContribution corresponding to the jacobian snapshots stored
in the [`ArrayContribution`](@ref) `c`. In transient problems, this procedure is
run for every order of the time derivative

"""
function reduced_jacobian(red::AbstractReduction,op,c::ArrayContribution)
  a,trians = map(get_domains(c),get_values(c)) do trian,values
    reduced_jacobian(red,op,values,trian)
  end |> tuple_of_arrays
  return Contribution(a,trians)
end

function reduced_jacobian_residual(solver::RBSolver,op,s)
  jac,res = jacobian_and_residual(solver,op,s)
  red_jac = reduced_jacobian(get_jacobian_reduction(solver),op,jac)
  red_res = reduced_residual(get_residual_reduction(solver),op,res)
  return red_jac,red_res
end

# ONLINE PHASE

function expand_caches!(a::AffineDecomposition,b::AbstractParamArray)
  coeff = a.coefficient
  result = a.result
  @check param_length(coeff) == param_length(result)
  param_length(coeff) == param_length(b) && return
  expand_cache!(coeff,param_length(b))
  expand_cache!(result,param_length(b))
  return
end

function expand_cache!(cache,plength)
  cache .= similar(coeff,eltype(cache),innersize(cache)...,plength)
  return
end

"""
    coefficient!(a::AffineDecomposition,b::AbstractParamArray) -> AbstractParamArray

Computes the MDEIM coefficient corresponding to the interpolated basis stored in
`a`, with respect to the interpolated snapshots `b`

"""
function coefficient!(a::AffineDecomposition,b::AbstractParamArray)
  coefficient = a.coefficient
  interpolation = a.interpolation
  ldiv!(coefficient,interpolation,b)
end

"""
    mdeim_result(a::AffineDecomposition,b::AbstractParamArray) -> AbstractParamArray

Returns the linear combination of the affine basis by the interpolated coefficient

"""
function mdeim_result(a::AffineDecomposition,b::AbstractParamArray)
  expand_caches!(a,b)
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

function allocate_result(red::AbstractReduction,test::MultiFieldRBSpace)
  active_block_ids = get_touched_blocks(test)
  block_result = [allocate_result(red,test[i]) for i in active_block_ids]
  return mortar(block_result)
end

function allocate_result(red::AbstractReduction,trial::MultiFieldRBSpace,test::MultiFieldRBSpace)
  active_block_ids = Iterators.product(get_touched_blocks(test),get_touched_blocks(trial))
  block_result = [allocate_result(red,trial[j],test[i]) for (i,j) in active_block_ids]
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
  red::AbstractReduction,
  op,
  s::BlockSnapshots,
  trian::Triangulation)

  test = get_test(op)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  ads,red_trians = [
    reduced_form(red,s[i],trian,test[i]) for i in active_block_ids
    ] |> tuple_of_arrays
  red_trian = ParamDataStructures.merge_triangulations(red_trians)
  cache = allocate_result(solver,test)
  ad = BlockAffineDecomposition(block_map,ads,cache)
  return ad,red_trian
end

function reduced_jacobian(
  red::AbstractReduction,
  op,
  s::BlockSnapshots,
  trian::Triangulation)

  trial = get_trial(op)
  test = get_test(op)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  ads,red_trians = [reduced_form(red,s[i,j],trian,trial[j],test[i])
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
