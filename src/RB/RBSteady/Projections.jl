"""
    abstract type Projection end

Represents a basis for a vector (sub)space, used as a (Petrov)-Galerkin projection
operator. In other words, Projection variables are operators from a high dimensional
vector space to a low dimensional one

Subtypes:
- [`SteadyProjection`](@ref)
- [`TransientProjection`](@ref)
- [`ReducedAlgebraicOperator`](@ref)

"""
abstract type Projection end

get_basis_space(a::Projection) = @abstractmethod
num_space_dofs(a::Projection) = @abstractmethod
num_reduced_space_dofs(a::Projection) = @abstractmethod
num_fe_dofs(a::Projection) = @abstractmethod
num_reduced_dofs(a::Projection) = @abstractmethod

"""
    abstract type SteadyProjection <: Projection end

Specialization for projections in steady problems. A constructor is given by
the function Projection

Subtypes:
- [`PODBasis`](@ref)
- [`TTSVDCores`](@ref)

"""
abstract type SteadyProjection <: Projection end

num_fe_dofs(a::SteadyProjection) = num_space_dofs(a)
num_reduced_dofs(a::SteadyProjection) = num_reduced_space_dofs(a)

function Projection(s::UnfoldingSteadySnapshots,args...;kwargs...)
  basis = tpod(s,args...;kwargs...)
  PODBasis(basis)
end

function Projection(s::UnfoldingSparseSnapshots,args...;kwargs...)
  basis = tpod(s,args...;kwargs...)
  sparse_basis = recast(s,basis)
  PODBasis(sparse_basis)
end

function Projection(s::AbstractSteadySnapshots,args...;kwargs...)
  cores = ttsvd(s,args...;kwargs...)
  index_map = get_index_map(s)
  TTSVDCores(cores,index_map)
end

function Projection(s::SparseSnapshots,args...;kwargs...)
  cores = ttsvd(s,args...;kwargs...)
  cores′ = recast(s,cores)
  index_map = get_index_map(s)
  TTSVDCores(cores′,index_map)
end

"""
    recast(x̂::AbstractVector,a::Projection) -> AbstractVector

Returns the action of the transposed Projection operator `a` on `x̂`

"""
function ParamDataStructures.recast(x̂::AbstractVector,a::SteadyProjection)
  basis = get_basis_space(a)
  x = basis*x̂
  return x
end

"""
    struct PODBasis{A<:AbstractMatrix} <: SteadyProjection

SteadyProjection stemming from a truncated proper orthogonal decomposition [`tpod`](@ref)

"""
struct PODBasis{A<:AbstractMatrix} <: SteadyProjection
  basis::A
end

get_basis_space(a::PODBasis) = a.basis
num_space_dofs(a::PODBasis) = size(get_basis_space(a),1)
num_reduced_space_dofs(a::PODBasis) = size(get_basis_space(a),2)

# TT interface

"""
    TTSVDCores{D,T,A<:AbstractVector{<:AbstractArray{T,D}},I} <: SteadyProjection

SteadyProjection stemming from a tensor train singular value decomposition [`ttsvd`](@ref).
An index map of type `I` is provided for indexing purposes

"""
struct TTSVDCores{D,T,A<:AbstractVector{<:AbstractArray{T,D}},I} <: SteadyProjection
  cores::A
  index_map::I
end

IndexMaps.get_index_map(a::TTSVDCores) = a.index_map

get_cores(a::TTSVDCores) = a.cores
get_spatial_cores(a::TTSVDCores) = a.cores

get_basis_space(a::TTSVDCores) = cores2basis(get_index_map(a),get_spatial_cores(a)...)
num_space_dofs(a::TTSVDCores) = prod(_num_tot_space_dofs(a))
num_reduced_space_dofs(a::TTSVDCores) = size(last(get_spatial_cores(a)),3)

_num_tot_space_dofs(a::TTSVDCores{3}) = size.(get_spatial_cores(a),2)

function _num_tot_space_dofs(a::TTSVDCores{4})
  scores = get_spatial_cores(a)
  tot_ndofs = zeros(Int,2,length(scores))
  @inbounds for i = eachindex(scores)
    tot_ndofs[:,i] .= size(scores[i],2),size(scores[i],3)
  end
  return tot_ndofs
end

"""
    cores2basis(index_map::AbstractIndexMap,cores::AbstractArray...) -> AbstractMatrix

Computes the kronecker product of the suitably indexed input cores

"""
function cores2basis(index_map::AbstractIndexMap,cores::AbstractArray...)
  cores2basis(_cores2basis(index_map,cores...))
end

function cores2basis(cores::AbstractArray...)
  c2m = _cores2basis(cores...)
  return dropdims(c2m;dims=1)
end

function cores2basis(core::AbstractArray{T,3}) where T
  pcore = permutedims(core,(2,1,3))
  return reshape(pcore,size(pcore,1),:)
end

function _cores2basis(a::AbstractArray{S,3},b::AbstractArray{T,3}) where {S,T}
  @check size(a,3) == size(b,1)
  TS = promote_type(T,S)
  nrows = size(a,2)*size(b,2)
  ab = zeros(TS,size(a,1),nrows,size(b,3))
  for i = axes(a,1), j = axes(b,3)
    for α = axes(a,3)
      @inbounds @views ab[i,:,j] += kronecker(b[α,:,j],a[i,:,α])
    end
  end
  return ab
end

function _cores2basis(a::AbstractArray{S,N},b::AbstractArray{T,N}) where {S,T,N}
  @abstractmethod
end

function _cores2basis(a::AbstractArray,b::AbstractArray...)
  c,d... = b
  return _cores2basis(_cores2basis(a,c),d...)
end

function _cores2basis(i::AbstractIndexMap,a::AbstractArray{T,3}...) where T
  basis = _cores2basis(a...)
  invi = inv_index_map(i)
  return view(basis,:,vec(invi),:)
end

# multi field interface

"""
    struct BlockProjection{A,N} <: AbstractArray{Projection,N} end

Block container for Projection of type `A` in a MultiField setting. This
type is conceived similarly to [`ArrayBlock`](@ref) in [`Gridap`](@ref)

"""
struct BlockProjection{A,N} <: AbstractArray{Projection,N}
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

Base.size(a::BlockProjection,i...) = size(a.array,i...)

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

function get_touched_blocks(a::BlockProjection)
  findall(a.touched)
end

function get_basis_space(a::BlockProjection{A,N}) where {A,N}
  basis_space = Array{Matrix{Float64},N}(undef,size(a))
  touched = a.touched
  for i in eachindex(a)
    if touched[i]
      basis_space[i] = get_basis_space(a[i])
    end
  end
  return ArrayBlock(basis_space,a.touched)
end

function num_space_dofs(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_space_dofs(a[i])
    end
  end
  return dofs
end

function num_reduced_space_dofs(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_reduced_space_dofs(a[i])
    end
  end
  return dofs
end

function Projection(s::BlockSnapshots;kwargs...)
  norm_matrix = fill(nothing,size(s))
  reduced_basis(s,norm_matrix;kwargs...)
end

function Projection(s::BlockSnapshots,norm_matrix;kwargs...)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  bases = [reduced_basis(s[i],norm_matrix[Block(i,i)];kwargs...) for i in active_block_ids]
  BlockProjection(block_map,bases)
end

"""
    enrich_basis(
      a::BlockProjection,
      norm_matrix::BlockMatrix,
      supr_op::BlockMatrix) -> BlockProjection

Returns the supremizer-enriched BlockProjection. This function stabilizes Inf-Sup
problems projected on a reduced vector space

"""
function enrich_basis(a::BlockProjection{<:PODBasis},norm_matrix::BlockMatrix,supr_op::BlockMatrix)
  bases = add_space_supremizers(get_basis_space(a),norm_matrix,supr_op)
  return BlockProjection(bases,a.touched)
end

"""
    add_space_supremizers(
      basis_space::MatrixBlock,
      norm_matrix::BlockMatrix,
      supr_op::BlockMatrix) -> Vector{<:Matrix}

Enriches the spatial basis `basis_space` with spatial supremizers computed from
the action of the supremizing operator `supr_op` on the dual field(s)

"""
function add_space_supremizers(basis_space,norm_matrix::BlockMatrix,supr_op::BlockMatrix)
  basis_primal,basis_dual... = basis_space.array
  A = norm_matrix[Block(1,1)]
  H = cholesky(A)
  for i = eachindex(basis_dual)
    b_i = supr_op[Block(1,i+1)] * basis_dual[i]
    supr_i = H \ b_i
    gram_schmidt!(supr_i,basis_primal,A)
    basis_primal = hcat(basis_primal,supr_i)
  end
  return [basis_primal,basis_dual...]
end
