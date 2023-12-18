function PartitionedArrays.allocate_local_values(
  a::PTArray,
  ::Type{T},
  indices) where T

  map(a) do ai
    similar(ai,T,local_length(indices))
  end
end

function PartitionedArrays.allocate_local_values(::Type{<:PTArray},indices)
  @notimplemented "The length of the PTArray is needed"
end

function PartitionedArrays.own_values(values::PTArray,indices)
  map(a->own_values(a,indices),values)
end

function PartitionedArrays.ghost_values(values::PTArray,indices)
  map(a->ghost_values(a,indices),values)
end

function PartitionedArrays.p_vector_cache_impl(
  ::Type{<:PTArray},
  vector_partition,
  index_partition)

  neighbors_snd,neighbors_rcv = assembly_neighbors(index_partition)
  indices_snd,indices_rcv = assembly_local_indices(
      index_partition,neighbors_snd,neighbors_rcv)
  buffers_snd,buffers_rcv = map(PartitionedArrays.assembly_buffers,
    vector_partition,indices_snd,indices_rcv) |> tuple_of_arrays
  map(PTVectorAssemblyCache,
    neighbors_snd,neighbors_rcv,indices_snd,indices_rcv,buffers_snd,buffers_rcv)
end

function PartitionedArrays.p_sparse_matrix_cache_impl(
  ::Type{<:PTArray},
  matrix_partition,
  row_partition,
  col_partition)

  neighbors_snd,neighbors_rcv = assembly_neighbors(index_partition)
  indices_snd,indices_rcv = assembly_local_indices(
      index_partition,neighbors_snd,neighbors_rcv)
  buffers_snd,buffers_rcv = map(PartitionedArrays.assembly_buffers,
    vector_partition,indices_snd,indices_rcv) |> tuple_of_arrays
  map(PTVectorAssemblyCache,
    neighbors_snd,neighbors_rcv,indices_snd,indices_rcv,buffers_snd,buffers_rcv)
end

function PartitionedArrays.assembly_buffers(
  values::PTArray,local_indices_snd,local_indices_rcv)

  T = eltype(values)
  N = length(values)
  ptrs = local_indices_snd.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = ptzeros(data,N)
  buffer_snd = PTJaggedArray(ptdata,ptrs)
  ptrs = local_indices_rcv.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = ptzeros(data,N)
  buffer_rcv = PTJaggedArray(ptdata,ptrs)
  buffer_snd,buffer_rcv
end

struct PTJaggedArray{T,Ti} <: AbstractVector{SubArray{T,1,Vector{T},Tuple{UnitRange{Ti}},true}}
  data::PTArray{T}
  ptrs::Vector{Ti}

  function PTJaggedArray(data::PTArray{T},ptrs::Vector{Ti}) where {T,Ti}
    new{T,Ti}(data,ptrs)
  end
  function PTJaggedArray{T,Ti}(data::PTArray{T},ptrs::Vector) where {T,Ti}
    new{T,Ti}(data,convert(Vector{Ti},ptrs))
  end
end

PartitionedArrays.JaggedArray(a::AbstractArray{<:PTArray{T}}) where T = JaggedArray{T,Int32}(a)
PartitionedArrays.JaggedArray(a::PTJaggedArray) = a
PartitionedArrays.JaggedArray{T,Ti}(a::PTJaggedArray{T,Ti}) where {T,Ti} = a

function PartitionedArrays.JaggedArray{T,Ti}(a::AbstractArray{<:PTArray{T}}) where {T,Ti}
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
  PTJaggedArray(data,ptrs)
end


Base.size(a::PTJaggedArray) = (length(a.ptrs)-1,)

function Base.getindex(a::PTJaggedArray,i::Int)
  map(a.data) do data
    getindex(JaggedArray(data,a.ptrs),i)
  end
end

function Base.setindex!(a::PTJaggedArray,v,i::Int)
  @notimplemented "Iterate over the inner jagged arrays instead"
end

function Base.show(io::IO,a::PTJaggedArray{A,B}) where {A,B}
  print(io,"PTJaggedArray{$A,$B}")
end

PartitionedArrays.jagged_array(data::PTArray,ptrs::Vector) = PTJaggedArray(data,ptrs)

struct PTVectorAssemblyCache{T}
  neighbors_snd::Vector{Int32}
  neighbors_rcv::Vector{Int32}
  local_indices_snd::JaggedArray{Int32,Int32}
  local_indices_rcv::JaggedArray{Int32,Int32}
  buffer_snd::PTJaggedArray{T,Int32}
  buffer_rcv::PTJaggedArray{T,Int32}
end

function Base.reverse(a::PTVectorAssemblyCache)
  PTVectorAssemblyCache(
                  a.neighbors_rcv,
                  a.neighbors_snd,
                  a.local_indices_rcv,
                  a.local_indices_snd,
                  a.buffer_rcv,
                  a.buffer_snd)
end
function PartitionedArrays.copy_cache(a::PTVectorAssemblyCache)
  buffer_snd = JaggedArray(copy(a.buffer_snd.data),a.buffer_snd.ptrs)
  buffer_rcv = JaggedArray(copy(a.buffer_rcv.data),a.buffer_rcv.ptrs)
  PTVectorAssemblyCache(
                  a.neighbors_snd,
                  a.neighbors_rcv,
                  a.local_indices_snd,
                  a.local_indices_rcv,
                  buffer_snd,
                  buffer_rcv)
end

Base.length(a::LocalView{T,N,<:PTArray}) where {T,N} = length(a.plids_to_value)
Base.size(a::LocalView{T,N,<:PTArray}) where {T,N} = (length(a),)

function Base.getindex(a::LocalView{T,N,<:PTArray},i::Integer...) where {T,N}
  LocalView(a.plids_to_value[i...],a.d_to_lid_to_plid)
end
