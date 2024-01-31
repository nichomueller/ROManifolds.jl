struct ParamJaggedArray{T,Ti,A,L} <: AbstractParamContainer{JaggedArray{T,Ti},1}
  data::ParamVector{T,A,L}
  ptrs::Vector{Ti}

  function ParamJaggedArray(data::ParamVector{T,A,L},ptrs::Vector{Ti}) where {T,Ti,A,L}
    new{T,Ti,A,L}(data,ptrs)
  end
  function ParamJaggedArray{T,Ti}(data::ParamVector{T,A,L},ptrs::Vector) where {T,Ti,A,L}
    new{T,Ti,A,L}(data,convert(Vector{Ti},ptrs))
  end
end

function PartitionedArrays.JaggedArray(data::ParamArray,ptrs)
  ParamJaggedArray(data,ptrs)
end

PartitionedArrays.JaggedArray(a::AbstractArray{<:ParamArray{T}}) where T = JaggedArray{T,Int32}(a)
PartitionedArrays.JaggedArray(a::ParamJaggedArray) = a
PartitionedArrays.JaggedArray{T,Ti}(a::ParamJaggedArray{T,Ti}) where {T,Ti} = a

Base.length(a::ParamJaggedArray) = length(a.data)
Base.size(a::ParamJaggedArray) = (length(a.ptrs)-1,)

function Base.getindex(a::ParamJaggedArray,index::Int)
  JaggedArray(a.data[index],a.ptrs)
end

function Base.setindex!(a::ParamJaggedArray,v,index::Int)
  @assert size(a) == size(v)
  setindex!(a.data,v,index)
end

# function Base.getindex(a::ParamJaggedArray{T,Ti,A,L},index::Int) where {T,Ti,A,L}
#   u = one(Ti)
#   pini = a.ptrs[index]
#   pend = a.ptrs[index+1]-u
#   ai = view(a.data[1],pini:pini)
#   b = Vector{typeof(ai)}(undef,L)
#   @inbounds for i = 1:L
#     b[i] = view(a.data[i],pini:pend)
#   end
#   ParamContainer(b)
# end

# function Base.setindex!(a::ParamJaggedArray{T,Ti,A,L},v,index::Int) where {T,Ti,A,L}
#   @assert length(a.data) == length(v)
#   u = one(Ti)
#   pini = a.ptrs[index]
#   pend = a.ptrs[index+1]-u
#   @inbounds for i = 1:L
#     a.data[i][pini:pend] = v[i]
#   end
# end

PartitionedArrays.jagged_array(data::ParamArray,ptrs::Vector) = ParamJaggedArray(data,ptrs)

struct ParamVectorAssemblyCache{T}
  neighbors_snd::Vector{Int32}
  neighbors_rcv::Vector{Int32}
  local_indices_snd::JaggedArray{Int32,Int32}
  local_indices_rcv::JaggedArray{Int32,Int32}
  buffer_snd::ParamJaggedArray{T,Int32}
  buffer_rcv::ParamJaggedArray{T,Int32}
end

function PartitionedArrays.VectorAssemblyCache(
  neighbors_snd,
  neighbors_rcv,
  local_indices_snd,
  local_indices_rcv,
  buffer_snd::ParamJaggedArray{T,Int32},
  buffer_rcv::ParamJaggedArray{T,Int32}) where T

  ParamVectorAssemblyCache(
    neighbors_snd,
    neighbors_rcv,
    local_indices_snd,
    local_indices_rcv,
    buffer_snd,
    buffer_rcv)
end

function Base.reverse(a::ParamVectorAssemblyCache)
  VectorAssemblyCache(
    a.neighbors_rcv,
    a.neighbors_snd,
    a.local_indices_rcv,
    a.local_indices_snd,
    a.buffer_rcv,
    a.buffer_snd)
end

function PartitionedArrays.copy_cache(a::ParamVectorAssemblyCache)
  buffer_snd = JaggedArray(copy(a.buffer_snd.data),a.buffer_snd.ptrs)
  buffer_rcv = JaggedArray(copy(a.buffer_rcv.data),a.buffer_rcv.ptrs)
  VectorAssemblyCache(
    a.neighbors_snd,
    a.neighbors_rcv,
    a.local_indices_snd,
    a.local_indices_rcv,
    buffer_snd,
    buffer_rcv)
end

struct ParamJaggedArrayAssemblyCache{T}
  cache::ParamVectorAssemblyCache{T}
end

function JaggedArrayAssemblyCache(cache::ParamVectorAssemblyCache{T}) where T
  ParamJaggedArrayAssemblyCache(cache)
end
