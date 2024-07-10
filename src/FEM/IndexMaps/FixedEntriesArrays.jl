struct FixedEntriesArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  array::A
  fixed_entries::Vector{NTuple{N,Int}}
end

Arrays.get_array(a::FixedEntriesArray) = a.array
get_fixed_entries(a::FixedEntriesArray) = a.fixed_entries

function FixedEntriesArray(array::AbstractArray,fixed_entries::AbstractVector{<:CartesianIndex})
  FixedEntriesArray(array,Tuple.(fixed_entries))
end

function FixedEntriesArray(array::AbstractArray,fixed_entries::AbstractVector{<:Integer})
  FixedEntriesArray(array,CartesianIndices(size(array))[fixed_entries])
end

function FixedEntriesArray(array::AbstractArray,fixed_entries::Integer)
  FixedEntriesArray(array,[fixed_entries])
end

Base.size(a::FixedEntriesArray) = size(a.array)

Base.@propagate_inbounds function Base.getindex(a::FixedEntriesArray{T,N},i::Vararg{Integer,N}) where {T,N}
  i ∈ a.fixed_entries ? zero(T) : getindex(a.array,i...)
end

Base.@propagate_inbounds function Base.setindex!(a::FixedEntriesArray{T,N},v,i::Vararg{Integer,N}) where {T,N}
  !(i ∈ a.fixed_entries) && setindex!(a.array,v,i...)
end

Base.copy(a::FixedEntriesArray) = FixedEntriesArray(copy(a.array),a.fixed_entries)

function Base.reshape(a::FixedEntriesArray,s::Vararg{Int})
  array′ = reshape(a.array,s...)
  indices = findall(array′.==zero(eltype(array′)))
  FixedEntriesArray(array′,indices)
end

function Base.stack(a::AbstractArray{<:FixedEntriesArray})
  array = stack(get_array.(a))
  indices = findall(array.==zero(eltype(array)))
  FixedEntriesArray(array,indices)
end
