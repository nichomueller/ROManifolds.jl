function IndexMaps.recast(a::AbstractVector{<:AbstractArray{T,3}},i::SparseIndexMap) where T
  us = IndexMaps.get_univariate_sparsity(i)
  @check length(us) ≤ length(a)
  if length(us) == length(a)
    return map(SparseCore,a,us)
  else
    asparse = map(i ->SparseCore(a[i],us[i]),eachindex(us))
    afull = a[length(us)+1:end]
    return [asparse...,afull...]
  end
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

num_space_dofs(a::SparseCoreCSC) = IndexMaps.num_rows(a.sparsity)*IndexMaps.num_cols(a.sparsity)

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

function block_cores(a::AbstractVector{<:AbstractArray}...)
  D = length(a)
  abfirst = first_block(getindex.(a,1)...)
  ablasts = map(d -> block_core(getindex.(a,d)...),2:D)
  return [abfirst,ablasts...]
end
