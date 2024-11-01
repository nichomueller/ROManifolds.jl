const ConsecutiveParamArrays{T} = Union{ConsecutiveArrayOfArrays{T},MatrixOfSparseMatricesCSC{T}}

Base.@propagate_inbounds function consecutive_getindex(A::ConsecutiveParamArrays,i...)
  @boundscheck checkbounds(A.data,i...)
  A.data[i...]
end

Base.@propagate_inbounds function consecutive_setindex!(A::ConsecutiveParamArrays,v,i...)
  @boundscheck checkbounds(A.data,i...)
  A.data[i...] = v
end

function consecutive_mul(A::AbstractArray,B::AbstractArray)
  A*B
end

function consecutive_mul(A::ConsecutiveParamArrays{T},B::ConsecutiveParamArrays{S}) where {T,S}
  A.data*B.data
end

function consecutive_mul(A::ConsecutiveParamArrays{T},B::Adjoint{S,<:ConsecutiveParamArrays}) where {T,S}
  A.data*adjoint(B.parent.data)
end

function consecutive_mul(A::Adjoint{T,<:ConsecutiveParamArrays},B::ConsecutiveParamArrays{S}) where {T,S}
  adjoint(A.parent.data)*B.data
end

function consecutive_mul(A::ConsecutiveParamArrays{T},B::Union{<:AbstractArray{S},Adjoint{S,<:AbstractArray}}) where {T,S}
  A.data*B
end

function consecutive_mul(A::Union{<:AbstractArray{T},Adjoint{T,<:AbstractArray}},B::ConsecutiveParamArrays{S}) where {T,S}
  A*B.data
end

# hacky
function Base.setindex!(
  A::ConsecutiveParamArrays,
  v::ConsecutiveParamArrays,
  i::Vararg{Integer}
  )

  A.data[i...,:] = v.data
  v
end
