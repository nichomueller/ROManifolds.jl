function IndexMaps.recast(a::AbstractVector{<:AbstractArray{T,3}},i::SparseIndexMap) where T
  N = length(a)
  us = IndexMaps.get_univariate_sparsity(i)
  @check length(us) ≤ N
  a′ = Vector{AbstractArray{T,3}}(undef,N)
  for n in eachindex(a)
    a′[n] = n ≤ length(us) ? SparseCore(a[n],us[n]) : a[n]
  end
  return a′
end

"""
    abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

Type for nonstandard representations of tensor train cores.

Subtypes:
- [`SparseCore`](@ref)

"""
abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

"""
    abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

Tensor train cores for sparse matrices.

Subtypes:
- [`SparseCoreCSC`](@ref)

"""
abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

"""
    struct SparseCoreCSC{T,Ti} <: SparseCore{T,3} end

Tensor train cores for sparse matrices in CSC format

"""
struct SparseCoreCSC{T,Ti} <: SparseCore{T,3}
  array::Array{T,3}
  sparsity::SparsityPatternCSC{T,Ti}
end

function SparseCore(array::Array{T,3},sparsity::SparsityPatternCSC{T}) where T
  SparseCoreCSC(array,sparsity)
end

Base.size(a::SparseCoreCSC) = size(a.array)
Base.getindex(a::SparseCoreCSC,i::Vararg{Integer,3}) = getindex(a.array,i...)
IndexMaps.get_sparsity(a::SparseCoreCSC) = get_sparsity(a.sparsity)

num_space_dofs(a::SparseCoreCSC) = IndexMaps.num_rows(a.sparsity)*IndexMaps.num_cols(a.sparsity)

to_4d_core(a::SparseCoreCSC) = SparseCoreCSC4D(a)

struct SparseCoreCSC4D{T,Ti} <: SparseCore{T,4}
  core::SparseCoreCSC{T,Ti}
  sparse_indexes::Vector{CartesianIndex{2}}
end

function SparseCoreCSC4D(core::SparseCoreCSC)
  sparsity = get_sparsity(core)
  irows,icols,_ = findnz(sparsity)
  SparseCoreCSC4D(core,CartesianIndex.(irows,icols))
end

Base.size(a::SparseCoreCSC4D) = (size(a.core.array,1),IndexMaps.num_rows(a.core.sparsity),
  IndexMaps.num_cols(a.core.sparsity),size(a.core.array,3))

function Base.getindex(a::SparseCoreCSC4D,i::Vararg{Integer,4})
  if CartesianIndex(i[2:3]) ∈ a.sparse_indexes
    i2 = findfirst(a.sparse_indexes .== [CartesianIndex(i[2:3])])
    i1,i3 = i[1],i[4]
    getindex(a.core.array,i1,i2,i3)
  else
    zero(eltype(a))
  end
end

# block cores

function first_block(a::AbstractArray{T,3},b::AbstractArray{S,3}) where {T,S}
  @check size(a,1) == size(b,1) == 1
  @check size(a,2) == size(b,2) "Cannot sum the two input cores"
  TS = promote_type(T,S)
  r2 = size(a,2)
  r3 = size(a,3) + size(b,3)
  ab = zeros(TS,1,r2,r3)
  @views ab[:,:,1:size(a,3)] = a
  @views ab[:,:,1+size(a,3):end] = b
  return ab
end

function last_block(a::AbstractArray{T,3},b::AbstractArray{S,3}) where {T,S}
  @check size(a,3) == size(b,3) == 1
  @check size(a,2) == size(b,2) "Cannot sum the two input cores"
  TS = promote_type(T,S)
  r1 = size(a,1) + size(b,1)
  r2 = size(a,2)
  ab = zeros(TS,r1,r2,1)
  @views ab[1:size(a,1),:,:] = a
  @views ab[1+size(a,1):end,:,:] = b
  return ab
end

function block_core(a::AbstractArray{T,3},b::AbstractArray{S,3}) where {T,S}
  @check size(a,2) == size(b,2) "Cannot sum the two input cores"
  TS = promote_type(T,S)
  r1 = size(a,1) + size(b,1)
  r2 = size(a,2)
  r3 = size(a,3) + size(b,3)
  ab = zeros(TS,r1,r2,r3)
  @views ab[1:size(a,1),:,1:size(a,3)] = a
  @views ab[1+size(a,1):end,:,1+size(a,3):end] = b
  return ab
end

for f in (:first_block,:block_core)
  @eval begin
    function $f(a::AbstractArray{T,3},b::AbstractArray...) where T
      bfirst,blasts... = b
      $f($f(a,bfirst),blasts...)
    end
  end
end

function block_cores(a::AbstractVector{<:AbstractArray{T}}...)::Vector{Array{T,3}} where T
  D = length(first(a))
  @check all(length(ai)==D for ai in a)
  abfirst = first_block(getindex.(a,1)...)
  ablasts = map(d -> block_core(getindex.(a,d)...),2:D)
  return [abfirst,ablasts...]
end
