"""
    abstract type Projection end

Represents a basis for a vector (sub)space, used as a (Petrov)-Galerkin projection
operator. In other words, Projection variables are operators from a high dimensional
vector space to a low dimensional one

Subtypes:
- [`PODProjection`](@ref)
- [`TTSVDProjection`](@ref)
- [`ReducedAlgebraicOperator`](@ref)

"""
abstract type Projection <: Map end

get_basis(a::Projection) = @abstractmethod
num_fe_dofs(a::Projection) = @abstractmethod
num_reduced_dofs(a::Projection) = @abstractmethod
project(a::Projection,x::AbstractArray) = @abstractmethod
inv_project(a::Projection,x::AbstractArray) = @abstractmethod
rescale(op::Function,x::AbstractArray,b::Projection) = @abstractmethod
galerkin_projection(a::Projection,b::Projection) = @abstractmethod
galerkin_projection(a::Projection,b::Projection,c::Projection,args...) = @abstractmethod
empirical_interpolation(a::Projection) = @abstractmethod
gram_schmidt(a::Projection,b::Projection,args...) = gram_schmidt(get_basis(a),get_basis(b),args...)

Base.:+(a::Projection,b::Projection) = union(a,b)
Base.:-(a::Projection,b::Projection) = union(a,b)
Base.:*(a::Projection,b::Projection) = galerkin_projection(a,b)
Base.:*(a::Projection,b::Projection,c::Projection) = galerkin_projection(a,b,c)
Base.:*(a::Projection,x::AbstractArray) = inv_project(a,x)
Base.:*(x::AbstractArray,b::Projection) = rescale(*,x,b)

function Base.:*(b::Projection,y::ConsecutiveArrayOfArrays)
  item = zeros(num_reduced_dofs(b))
  plength = param_length(y)
  x = array_of_consecutive_arrays(item,plength)
  mul!(x,b,y)
end

function LinearAlgebra.mul!(x::AbstractArray,b::Projection,y::AbstractArray,α::Number,β::Number)
  mul!(x,get_basis(b),y,α,β)
end

function LinearAlgebra.mul!(x::ConsecutiveArrayOfArrays,b::Projection,y::ConsecutiveArrayOfArrays,α::Number,β::Number)
  mul!(x.data,get_basis(b),y.data,α,β)
end

# row space to column space
function project(a::Projection,x::AbstractVector)
  basis = get_basis(a)
  x̂ = basis'*x
  return x̂
end

# column space to row space
function inv_project(a::Projection,x̂::AbstractVector)
  basis = get_basis(a)
  x = basis*x̂
  return x
end

function Arrays.return_cache(::typeof(project),a::Projection,x::AbstractVector)
  x̂ = zeros(eltype(x),num_reduced_dofs(a))
  return x̂
end

function Arrays.return_cache(::typeof(inv_project),a::Projection,x̂::AbstractVector)
  x = zeros(eltype(x̂),num_fe_dofs(a))
  return x
end

struct InvProjection{P<:Projection} <: Projection
  projection::P
end

Base.adjoint(a::Projection) = InvProjection(a)

get_basis(a::InvProjection) = get_basis(a)
num_fe_dofs(a::InvProjection) = num_fe_dofs(a)
num_reduced_dofs(a::InvProjection) = num_reduced_dofs(a)
project(a::InvProjection,x::AbstractArray) = inv_project(a.projection,x)
inv_project(a::InvProjection,x::AbstractArray) = project(a.projection,x)

abstract type ReducedProjection{A<:AbstractArray} <: Projection end

const ReducedVecProjection = ReducedProjection{<:AbstractMatrix}
const ReducedMatProjection = ReducedProjection{<:AbstractArray{T,3} where T}

function project(a::ReducedProjection,x::AbstractVector)
  @notimplemented
end

function inv_project(a::ReducedMatProjection,x̂::AbstractVector)
  basis = get_basis(a)
  x = contraction(basis,x̂)
  return x
end

function LinearAlgebra.mul!(
  x::AbstractArray,
  b::ReducedMatProjection,
  y::AbstractArray,
  α::Number,β::Number)

  contraction!(x,get_basis(b),y,α,β)
end

function LinearAlgebra.mul!(
  x::ConsecutiveArrayOfArrays,
  b::ReducedMatProjection,
  y::ConsecutiveArrayOfArrays,
  α::Number,β::Number)

  contraction!(x.data,get_basis(b),y.data,α,β)
end

struct ReducedAlgebraicProjection{A} <: ReducedProjection{A}
  basis::A
end

ReducedProjection(basis::AbstractArray) = ReducedAlgebraicProjection(basis)

get_basis(a::ReducedAlgebraicProjection) = a.basis
num_reduced_dofs(a::ReducedAlgebraicProjection) = size(get_basis(a),2)
num_reduced_dofs_left_projector(a::ReducedAlgebraicProjection) = size(get_basis(a),1)
num_reduced_dofs_right_projector(a::ReducedMatProjection) = size(get_basis(a),3)

function projection(red::AffineReduction,s::AbstractMatrix,args...)
  podred = PODReduction(ReductionStyle(red),NormStyle(red))
  projection(podred,s,args...)
end

"""
    struct PODBasis{A<:AbstractMatrix} <: Projection

Projection stemming from a truncated proper orthogonal decomposition [`truncated_pod`](@ref)

"""
struct PODBasis{A<:AbstractMatrix} <: Projection
  basis::A
end

function projection(red::PODReduction,s::AbstractArray{<:Number},args...)
  basis = reduction(red,s,args...)
  PODBasis(basis)
end

get_basis(a::PODBasis) = a.basis
num_fe_dofs(a::PODBasis) = size(get_basis(a),1)
num_reduced_dofs(a::PODBasis) = size(get_basis(a),2)

function rescale(op::Function,x::AbstractArray,b::PODBasis)
  PODBasis(op(x,get_basis(b)))
end

Base.union(a::PODBasis,b::PODBasis,args...) = union(a,get_basis(b),args...)

function Base.union(a::PODBasis,basis_b::AbstractMatrix,args...)
  basis_a = get_basis(a)
  basis_ab, = gram_schmidt(basis_b,basis_a,args...)
  PODBasis(basis_ab)
end

function galerkin_projection(proj_left::PODBasis,a::PODBasis)
  basis_left = get_basis(proj_left)
  basis = get_basis(a)
  proj_basis = galerkin_projection(basis_left,basis)
  return ReducedProjection(proj_basis)
end

function galerkin_projection(proj_left::PODBasis,a::PODBasis,proj_right::PODBasis)
  basis_left = get_basis(proj_left)
  basis = get_basis(a)
  basis_right = get_basis(proj_right)
  proj_basis = galerkin_projection(basis_left,basis,basis_right)
  return ReducedProjection(proj_basis)
end

function empirical_interpolation(a::PODBasis)
  empirical_interpolation(get_basis(a))
end

# TT interface

"""
    TTSVDCores{A<:AbstractVector{<:AbstractArray{T,D}},I} <: Projection

Projection stemming from a tensor train singular value decomposition [`ttsvd`](@ref).
An index map of type `I` is provided for indexing purposes

"""
struct TTSVDCores{D,A<:AbstractVector{<:AbstractArray{T,3} where T},I<:AbstractIndexMap{D}} <: Projection
  cores::A
  index_map::I
end

function projection(red::TTSVDReduction,s::AbstractSnapshots,args...)
  cores = reduction(red,s,args...)
  index_map = get_index_map(s)
  TTSVDCores(cores,index_map)
end

get_cores(a::TTSVDCores) = a.cores

get_basis(a::TTSVDCores) = cores2basis(get_index_map(a),get_cores(a)...)
num_fe_dofs(a::TTSVDCores) = prod(map(c -> size(c,2),get_cores(a)))
num_reduced_dofs(a::TTSVDCores) = size(last(get_cores(a)),3)

IndexMaps.get_index_map(a::TTSVDCores) = a.index_map

function rescale(op::Function,x::AbstractRankTensor{D1},b::TTSVDCores{D2}) where {D1,D2}
  if D1 == D2
    TTSVDCores(op(x,get_cores(b)),get_index_map(b))
  else
    c1 = op(x,get_cores(b)[1:D1])
    c2 = get_cores(b)[D1+1:end]
    TTSVDCores([c1...,c2...],get_index_map(b))
  end
end

function Base.union(a::TTSVDCores,b::TTSVDCores,args...)
  @check get_index_map(a) == get_index_map(b)
  union(a,get_cores(b),args...)
end

function Base.union(a::TTSVDCores,cores_b::AbstractVector{<:AbstractArray},args...)
  cores_a = get_cores(a)
  cores_ab = block_cores(cores_a,cores_b)
  orthogonalize!(cores_ab,args...)
  TTSVDCores(cores_ab,get_index_map(a))
end

function galerkin_projection(proj_left::TTSVDCores,a::TTSVDCores)
  cores_left = get_cores(proj_left)
  cores = get_cores(a)
  proj_basis = galerkin_projection(cores_left,cores)
  return ReducedProjection(proj_basis)
end

function galerkin_projection(proj_left::TTSVDCores,a::TTSVDCores,proj_right::TTSVDCores)
  cores_left = get_cores(proj_left)
  cores = get_cores(a)
  cores_right = get_cores(proj_right)
  proj_basis = galerkin_projection(cores_left,cores,cores_right)
  return ReducedProjection(proj_basis)
end

function empirical_interpolation(a::TTSVDCores)
  cores = get_cores(a)
  index_map = get_index_map(a)
  c = cores2basis(first(cores))
  cache = eim_cache(c)
  vinds = Vector{Int32}[]
  for i = eachindex(cores)
    inds,interp = empirical_interpolation!(cache,c)
    push!(vinds,copy(inds))
    if i < length(cores)
      interp_core = reshape(interp,1,size(interp)...)
      c = cores2basis(interp_core,cores[i+1])
    else
      indices = basis_indices(vinds,index_map)
      return indices,interp
    end
  end
end

# multi field interface

function Arrays.return_cache(::typeof(projection),::PODReduction,s::AbstractSnapshots)
  b = testvalue(Matrix{eltype(s)})
  return PODBasis(b)
end

function Arrays.return_cache(::typeof(projection),::TTSVDReduction,s::AbstractSnapshots)
  c = testvalue(Vector{Array{eltype(s),3}})
  i = get_index_map(s)
  return TTSVDCores(c,i)
end

function Arrays.return_cache(::typeof(projection),red::Reduction,s::BlockSnapshots)
  i = findfirst(s.touched)
  @notimplementedif isnothing(i)
  basis = return_cache(projection,red,blocks(s)[i])
  block_basis = Array{typeof(basis),ndims(s)}(undef,size(s))
  touched = s.touched
  return BlockProjection(block_basis,touched)
end

function projection(red::Reduction,s::BlockSnapshots)
  basis = return_cache(projection,red,s)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i])
    end
  end
  return basis
end

function projection(red::Reduction,s::BlockSnapshots,norm_matrix)
  basis = return_cache(projection,red,s)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i],norm_matrix[Block(i,i)])
    end
  end
  return basis
end

"""
    struct BlockProjection{A,N} <: Projection end

Block container for Projection of type `A` in a MultiField setting. This
type is conceived similarly to [`ArrayBlock`](@ref) in [`Gridap`](@ref)

"""
struct BlockProjection{A<:Projection,N} <: Projection
  array::Array{A,N}
  touched::Array{Bool,N}

  function BlockProjection(
    array::Array{A,N},
    touched::Array{Bool,N}
    ) where {A<:Projection,N}

    @check size(array) == size(touched)
    new{A,N}(array,touched)
  end
end

function BlockProjection(k::BlockMap{N},a::AbstractArray{A}) where {A<:Projection,N}
  array = Array{A,N}(undef,k.size)
  touched = fill(false,k.size)
  for (t,i) in enumerate(k.indices)
    array[i] = a[t]
    touched[i] = true
  end
  BlockProjection(array,touched)
end

function BlockProjection(a::AbstractArray{A},touched::Array{Bool,N}) where {A<:Projection,N}
  array = Array{A,N}(undef,k.size)
  for i in touched
    if touched[i]
      array[i] = a[i]
    end
  end
  BlockProjection(array,touched)
end

Base.ndims(a::BlockProjection) = ndims(a.touched)
Base.size(a::BlockProjection) = size(a.touched)
Base.axes(a::BlockProjection) = axes(a.touched)
Base.eachindex(a::BlockProjection) = eachindex(a.touched)

function Base.getindex(a::BlockProjection,i...)
  if !a.touched[i...]
    return nothing
  end
  a.array[i...]
end

function Base.setindex!(a::BlockProjection,v,i...)
  @check a.touched[i...] "Only touched entries can be set"
  a.array[i...] = v
end

function get_basis(a::BlockProjection{A,N}) where {A,N}
  @notimplemented
end

function get_cores(a::BlockProjection{<:TTSVDCores,N}) where N
  @notimplemented
end

function num_fe_dofs(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_fe_dofs(a[i])
    end
  end
  return dofs
end

function num_reduced_dofs(a::BlockProjection)
  dofs = zeros(Int,size(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_reduced_dofs(a[i])
    end
  end
  return dofs
end

for f in (:project,:inv_project)
  @eval begin
    function Arrays.return_cache(
      ::typeof($f),
      a::BlockProjection,
      x::Union{BlockVector,BlockVectorOfVectors})

      @check size(a) == nblocks(x)
      y = Vector{eltype(x)}(undef,nblocks(a))
      for i in eachindex(a)
        if a.touched[i]
          y[Block(i)] = return_cache($f,a[i],blocks(x)[i])
        else
          y[Block(i)] = blocks(x)[i]
        end
      end
      return mortar(y)
    end

    function $f(a::BlockProjection,x::Union{BlockArray,BlockArrayOfArrays})
      y = return_cache($f,a,x)
      for i in eachindex(a)
        if a.touched[i]
          y[Block(i)] = $f(a[i],blocks(x)[i])
        end
      end
      return y
    end
  end
end

"""
    enrich!(
      a::BlockProjection,
      norm_matrix::AbstractMatrix,
      supr_matrix::AbstractMatrix,
      args...) -> BlockProjection

Returns the supremizer-enriched BlockProjection. This function stabilizes Inf-Sup
problems projected on a reduced vector space

"""
function enrich!(
  red::SupremizerReduction,
  a::BlockProjection,
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix)

  @check a.touched[1] "Primal field not defined"
  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = cholesky(X_primal)
  for i = eachindex(a_dual)
    dual_i = get_basis_space(a_dual[i])
    C_primal_dual_i = supr_matrix[Block(1,i+1)]
    supr_i = H_primal \ C_primal_dual_i * dual_i
    a_primal = union(a_primal,supr_i,X_primal)
  end
  a[1] = a_primal
  return
end
