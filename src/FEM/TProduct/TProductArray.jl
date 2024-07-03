"""
    abstract type AbstractTProductArray{T,N} <: AbstractVector{AbstractArray{T,N}} end

Type storing information related to FE arrays in a 1-D setting. The FE array is
defined as their permuted (see [`get_tp_dof_index_map`](@ref)) tensor product.

Subtypes:
- [`TProductArray`](@ref)
- [`TProductGradientArray`](@ref)
- [`TProductDivergenceArray`](@ref)
- [`BlockTProductArray`](@ref)

"""
abstract type AbstractTProductArray{T,N} <: AbstractVector{AbstractArray{T,N}} end

Arrays.get_array(a::AbstractTProductArray) = @abstractmethod
Base.size(a::AbstractTProductArray) = size(get_array(a))
Base.getindex(a::AbstractTProductArray,i::Integer) = getindex(get_array(a),i)

"""
    TProductArray{T,N,A} <: AbstractTProductArray{T,N}

Represents a mass matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    M₁₂₃... = U+1D4D8(M₁ ⊗ M₂ ⊗ M₃ ⊗ ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
U+1D4D8(⋅) is the index map, and M₁₂₃... is the D-dimensional mass matrix

"""
struct TProductArray{T,N,A<:AbstractArray{T,N}} <: AbstractTProductArray{T,N}
  arrays_1d::Vector{A}
end

function tproduct_array(arrays_1d::Vector{<:AbstractArray})
  TProductArray(arrays_1d)
end

Arrays.get_array(a::TProductArray) = a.arrays_1d

"""
    TProductGradientArray{T,N,A} <: AbstractTProductArray{T,N}

Represents a stiffness matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    A₁₂₃... = U+1D4D8(M₁ ⊗ M₂ ⊗ M₃ ⊗ ... + A₁ ⊗ M₂ ⊗ M₃ ⊗ ... + M₁ ⊗ A₂ ⊗ M₃ ⊗ ... + ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
A₁, A₂, A₃, ... represent the 1-D stiffness matrices on their respective axes,
U+1D4D8(⋅) is the index map, and A₁₂₃... is the D-dimensional stiffness matrix

"""
struct TProductGradientArray{T,N,A<:AbstractArray{T,N}} <: AbstractTProductArray{T,N}
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
end

function tproduct_array(::typeof(gradient),arrays_1d::Vector{<:AbstractArray},gradients_1d::Vector{<:AbstractArray})
  TProductGradientArray(arrays_1d,gradients_1d)
end

Arrays.get_array(a::TProductGradientArray) = a.arrays_1d

"""
    TProductDivergenceArray{T,N,A} <: AbstractTProductArray{T,N}

Represents a pressure-velocity matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    B₁₂₃... = U+1D4D8(B₁ ⊗ M₂ ⊗ M₃ ⊗ ... + M₁ ⊗ B₂ ⊗ M₃ ⊗ ... + ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
B₁, B₂, B₃, ... represent the 1-D pressure-velocity matrices on their respective axes,
U+1D4D8(⋅) is the index map, and B₁₂₃... is the D-dimensional pressure-velocity matrix

"""
struct TProductDivergenceArray{T,N,A<:AbstractArray{T,N}} <: AbstractTProductArray{T,N}
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
end

function tproduct_array(::typeof(divergence),arrays_1d::Vector{<:AbstractArray},gradients_1d::Vector{<:AbstractArray})
  TProductDivergenceArray(arrays_1d,gradients_1d)
end

Arrays.get_array(a::TProductDivergenceArray) = a.arrays_1d

# MultiField interface
struct BlockTProductArray{A<:AbstractTProductArray,N} <: AbstractArray{A,N}
  array::Array{A,N}
end

function tproduct_array(arrays_1d::Vector{<:BlockArray})
  nblocks = blocklength(first(arrays_1d))
  arrays = map(1:nblocks) do i
    arrays_1d_i = map(blocks,arrays_1d)[i]
    tproduct_array(arrays_1d_i)
  end
  BlockTProductArray(arrays)
end

function tproduct_array(arrays_1d::Vector{<:BlockArray},gradients_1d::Vector{<:BlockArray})
  nblocks = blocklength(first(arrays_1d))
  arrays = map(1:nblocks) do i
    arrays_1d_i = map(blocks,arrays_1d)[i]
    gradients_1d_i = map(blocks,gradients_1d)[i]
    tproduct_array(arrays_1d_i,gradients_1d_i)
  end
  BlockTProductArray(arrays)
end

Base.size(a::BlockTProductArray) = size(a.array)

Base.@propagate_inbounds function Base.getindex(
  a::BlockTProductArray{A,N},
  i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck blockcheckbounds(a.array,i)
  getindex(a.array,i...)
end

Base.@propagate_inbounds function Base.setindex!(
  a::BlockTProductArray{A,N},
  v,i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck blockcheckbounds(a.array,i)
  setindex!(a.array,v,i...)
end
