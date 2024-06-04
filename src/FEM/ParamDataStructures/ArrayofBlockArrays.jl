# struct BlockParamArray{T,N,A<:AbstractArray{<:ParamArray{T,N},N},B<:NTuple{N,AbstractUnitRange{Int}}} <: AbstractBlockArray{AbstractParamArray{T,N},N}
#   blocks::A
#   axes::B

#   function BlockParamArray(
#     blocks::A,
#     axes::B
#     ) where {A<:AbstractArray{<:ParamArray{T,N},N},B<:NTuple{N,AbstractUnitRange{Int}}}

#     new{A,B}(blocks,axes)
#   end
# end

# const BlockParamVector{T,A<:AbstractVector{<:ParamVector{T}}} = BlockParamArray{T,1,A}
# const BlockParamMatrix{T,A<:AbstractMatrix{<:ParamMatrix{T}}} = BlockParamArray{T,2,A}

# function BlockArrays.mortar(blocks::AbstractArray{<:ParamArray})
#   block_sizes = BlockArrays.sizes_from_blocks(first.(blocks))
#   block_axes = map(blockedrange,block_sizes)
#   BlockParamArray(blocks,block_axes)
# end

# Base.copy(a::BlockParamArray) = BlockParamArray(map(copy,a.blocks),a.axes)
# @inline Base.axes(a::BlockParamArray) = a.axes

# @inline function Base.similar(a::BlockParamArray,::Type{T}) where T
#   mortar(similar.(blocks(a),T))
# end

# @inline function Base.getindex(a::BlockParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
#   @boundscheck checkbounds(a,i...)
#   @inbounds v = a[findblockindex.(axes(a),i)...]
#   return v
# end

# @inline function Base.setindex!(a::BlockArray{T,N},v,i::Vararg{Integer,N}) where {T,N}
#   @boundscheck checkbounds(a,i...)
#   @inbounds a[findblockindex.(axes(a),i)...] = v
#   return a
# end
