"""
    struct Range2D{I<:AbstractVector,J<:AbstractVector} <: AbstractMatrix{Int}
      axis1::I
      axis2::J
      scale::Int
    end

Creates the indices of a Kronecker product matrix from the indices of the two factors.
The field `axis1` refers to the first factor, field `axis2` refers to the second
factor. The field `scale` is by default equal to the number of entries of the
first factor.
"""
struct Range2D{I<:AbstractVector,J<:AbstractVector} <: AbstractMatrix{Int}
  axis1::I
  axis2::J
  scale::Int
end

"""
    range_2d(i::AbstractVector,j::AbstractVector,scale=length(i)) -> Range2D

Constructor of a [`Range2D`](@ref) object
"""
range_2d(i::AbstractVector,j::AbstractVector,scale=length(i)) = Range2D(i,j,scale)

"""
    range_1d(i::AbstractVector,j::AbstractVector,args...) -> AbstractVector{Int}

Vectorization operation of a [`Range2D`](@ref) object
"""
range_1d(i::AbstractVector,j::AbstractVector,args...) = vec(range_2d(i,j,args...))

Base.size(r::Range2D) = (length(r.axis1),length(r.axis2))
Base.getindex(r::Range2D,i::Integer,j::Integer) = r.axis1[i] + (r.axis2[j]-1)*r.scale
