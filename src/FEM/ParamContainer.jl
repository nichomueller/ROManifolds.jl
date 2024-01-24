abstract type AbstractParamContainer{T,N} <: AbstractArray{T,N} end

struct ParamContainer{T,L} <: AbstractParamContainer{T,1}
  array::AbstractVector{T}
  function ParamContainer(array::AbstractVector{T},::Val{L}) where {T,L}
    new{T,L}(array)
  end
end

ParamContainer(array::AbstractVector{T}) where T = ParamContainer(array,Val(length(array)))

Base.length(c::ParamContainer{T,L}) where {T,L} = L
Base.size(c::ParamContainer) = (length(c),)
Base.getindex(c::ParamContainer,i::Integer) = c.array[i]
Base.iterate(c::ParamContainer,i...) = iterate(c.array,i...)
