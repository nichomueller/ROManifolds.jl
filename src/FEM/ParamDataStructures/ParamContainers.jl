struct ParamContainer{T} <: AbstractVector{T}
  array::Vector{T}
end

ParamContainer(a::AbstractArray{<:Number}) = a
ParamContainer(a::AbstractArray{<:AbstractArray}) = ArrayOfSimilarArrays(a)

param_length(a::ParamContainer) = length(a.array)
param_getindex(a::ParamContainer,i::Integer) = getindex(a.array,i)

Base.size(a::ParamContainer) = (param_length(a),)
Base.getindex(a::ParamContainer,i::Integer) = param_getindex(a,i)
