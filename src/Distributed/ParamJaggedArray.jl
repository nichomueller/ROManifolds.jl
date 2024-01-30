
struct ParamJaggedArray{T,Ti,A,L} <: AbstractVector{SubArray{T,1,ParamVector{T,A,L},Tuple{UnitRange{Ti}},true}}
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

function PartitionedArrays.JaggedArray{T,Ti}(a::AbstractArray{<:ParamArray{T}}) where {T,Ti}
  n = length(a)
  ptrs = Vector{Ti}(undef,n+1)
  u = one(eltype(ptrs))
  @inbounds for i in 1:n
    ai = a[i]
    ptrs[i+1] = length(ai)
  end
  length_to_ptrs!(ptrs)
  ndata = ptrs[end]-u
  data = Vector{T}(undef,ndata)
  p = 1
  @inbounds for i in 1:n
    ai = a[i]
    for j in eachindex(ai)
      aij = ai[j]
      data[p] = aij
      p += 1
    end
  end
  ParamJaggedArray(data,ptrs)
end


Base.size(a::ParamJaggedArray) = (length(a.ptrs)-1,)

function Base.getindex(a::ParamJaggedArray,i::Int)
  map(a.data) do data
    getindex(JaggedArray(data,a.ptrs),i)
  end
end

function Base.setindex!(a::ParamJaggedArray,v,i::Int)
  @notimplemented "Iterate over the inner jagged arrays instead"
end

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
