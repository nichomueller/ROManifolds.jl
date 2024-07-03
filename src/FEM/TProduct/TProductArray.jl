"""
    abstract type AbstractTProductArray end

Type storing information related to FE arrays in a 1-D setting. The FE array is
defined as their permuted (see [`get_tp_dof_index_map`](@ref)) tensor product.

Subtypes:
- [`TProductArray`](@ref)
- [`TProductGradientArray`](@ref)
- [`TProductDivergenceArray`](@ref)
- [`BlockTProductArray`](@ref)

"""
abstract type AbstractTProductArray end

get_arrays(a::AbstractTProductArray) = @abstractmethod
tp_length(a::AbstractTProductArray) = length(get_arrays(a))

function tp_getindex(a::AbstractTProductArray,d::Integer)
  D = tp_length(a)
  @check d ≤ D
  tp_getindex(a,Val{d}(),Val{D}())
end

"""
    TProductArray{T,N,A} <: AbstractTProductArray

Represents a mass matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    M₁₂₃... = U+1D4D8(M₁ ⊗ M₂ ⊗ M₃ ⊗ ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
U+1D4D8(⋅) is the index map, and M₁₂₃... is the D-dimensional mass matrix

"""
struct TProductArray{T,N,A<:AbstractArray{T,N}} <: AbstractTProductArray
  arrays_1d::Vector{A}
end

function tproduct_array(arrays_1d::Vector{<:AbstractArray})
  TProductArray(arrays_1d)
end

get_arrays(a::TProductArray) = a.arrays_1d

tp_getindex(a::TProductArray,::Val{d},::Val{D}) where {d,D} = get_arrays(a)[d]

"""
    TProductGradientArray{T,N,A} <: AbstractTProductArray

Represents a stiffness matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    A₁₂₃... = U+1D4D8(M₁ ⊗ M₂ ⊗ M₃ ⊗ ... + A₁ ⊗ M₂ ⊗ M₃ ⊗ ... + M₁ ⊗ A₂ ⊗ M₃ ⊗ ... + ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
A₁, A₂, A₃, ... represent the 1-D stiffness matrices on their respective axes,
U+1D4D8(⋅) is the index map, and A₁₂₃... is the D-dimensional stiffness matrix

"""
struct TProductGradientArray{T,N,A<:AbstractArray{T,N}} <: AbstractTProductArray
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
end

function tproduct_array(::typeof(gradient),arrays_1d::Vector{<:AbstractArray},gradients_1d::Vector{<:AbstractArray})
  TProductGradientArray(arrays_1d,gradients_1d)
end

get_arrays(a::TProductGradientArray) = a.arrays_1d
get_gradients(a::TProductGradientArray) = a.gradients_1d

tp_getindex(a::TProductGradientArray,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_getindex(a::TProductGradientArray,::Val{1},::Val{1}) = [get_arrays(a)[1],get_gradients(a)[1]]
tp_getindex(a::TProductGradientArray,::Val{1},::Val{2}) = [get_arrays(a)[1],get_gradients(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientArray,::Val{1},::Val{2}) = [get_arrays(a)[2],get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductGradientArray,::Val{1},::Val{3}) = [get_arrays(a)[1],get_gradients(a)[1],get_arrays(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientArray,::Val{2},::Val{3}) = [get_arrays(a)[2],get_arrays(a)[2],get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductGradientArray,::Val{3},::Val{3}) = [get_arrays(a)[3],get_arrays(a)[3],get_arrays(a)[3],get_gradients(a)[3]]

"""
    TProductDivergenceArray{T,N,A} <: AbstractTProductArray

Represents a pressure-velocity matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    B₁₂₃... = U+1D4D8(B₁ ⊗ M₂ ⊗ M₃ ⊗ ... + M₁ ⊗ B₂ ⊗ M₃ ⊗ ... + ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
B₁, B₂, B₃, ... represent the 1-D pressure-velocity matrices on their respective axes,
U+1D4D8(⋅) is the index map, and B₁₂₃... is the D-dimensional pressure-velocity matrix

"""
struct TProductDivergenceArray{T,N,A<:AbstractArray{T,N}} <: AbstractTProductArray
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
end

function tproduct_array(::typeof(divergence),arrays_1d::Vector{<:AbstractArray},gradients_1d::Vector{<:AbstractArray})
  TProductDivergenceArray(arrays_1d,gradients_1d)
end

get_arrays(a::TProductDivergenceArray) = a.arrays_1d
get_gradients(a::TProductDivergenceArray) = a.gradients_1d

tp_getindex(a::TProductDivergenceArray,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_getindex(a::TProductDivergenceArray,::Val{1},::Val{1}) = [get_gradients(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductDivergenceArray,::Val{1},::Val{2}) = [get_gradients(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductDivergenceArray,::Val{1},::Val{2}) = [get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductDivergenceArray,::Val{1},::Val{3}) = [get_gradients(a)[1],get_arrays(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductDivergenceArray,::Val{2},::Val{3}) = [get_gradients(a)[2],get_arrays(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductDivergenceArray,::Val{3},::Val{3}) = [get_gradients(a)[3],get_arrays(a)[3],get_arrays(a)[3]]

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

for T in (:(typeof(gradient)),:(typeof(divergence)))
  @eval begin
    function tproduct_array(op::$T,arrays_1d::Vector{<:BlockArray},gradients_1d::Vector{<:BlockArray})
      s_blocks = blocksize(first(arrays_1d))
      arrays = map(CartesianIndices(s_blocks)) do i
        iblock = Block(Tuple(i))
        arrays_1d_i = getindex.(arrays_1d,iblock)
        gradients_1d_i = getindex.(gradients_1d,iblock)
        tproduct_array(op,arrays_1d_i,gradients_1d_i)
      end
      BlockTProductArray(arrays)
    end
  end
end

Base.size(a::BlockTProductArray) = size(a.array)

Base.@propagate_inbounds function Base.getindex(
  a::BlockTProductArray{A,N},
  i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  getindex(a.array,i...)
end

Base.@propagate_inbounds function Base.setindex!(
  a::BlockTProductArray{A,N},
  v,i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  setindex!(a.array,v,i...)
end

Base.@propagate_inbounds function Base.getindex(a::BlockTProductArray{A,N},i::Block{N}) where {A,N}
  getindex(a.array,i.n...)
end
