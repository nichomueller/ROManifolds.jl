struct FixedEntriesArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  array::A
  fixed_entries::Vector{NTuple{N,Int}}
end

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
  i ∈ a.fixed_entries && setindex!(a.array,v,i...)
end

# function Base.view(a::FixedEntriesArray{T,N},i::Vararg{<:Any,N}) where {T,N}
#   iints = findall((!isa).(i,Integer))
#   fixed_entries = map(a->getindex(a,iints),a.array)
#   FixedEntriesArray(view(a.array,i...),fixed_entries)
# end
