# function PartitionedArrays.allocate_gather_impl(
#   snd::PTArray,
#   destination,
#   ::Type{T}) where T

#   function f(snd)
#     sndi = testitem(snd)
#     ni = length(sndi)
#     vi = Vector{T}(undef,ni)
#     return PTArray([copy(vi) for i = eachindex(snd)])
#   end
#   if isa(destination,Integer)
#     function g(snd)
#       vi = Vector{T}(undef,0)
#       return PTArray([copy(vi) for i = eachindex(snd)])
#     end
#     rcv = map_main(f,snd;otherwise=g,main=destination)
#   else
#     @assert destination === :all
#     rcv = map(f,snd)
#   end
#   rcv
# end

# function PartitionedArrays.allocate_gather_impl(
#   snd::PTArray,
#   destination,
#   ::Type{T}) where T<:AbstractVector

#   l = map(length,snd)
#   l_dest = gather(l;destination)
#   function f(l,snd)
#     ptrs = length_to_ptrs!(pushfirst!(l,one(eltype(l))))
#     ndata = ptrs[end]-1
#     data = Vector{eltype(snd)}(undef,ndata)
#     JaggedArray{eltype(snd),Int32}(data,ptrs)
#   end
#   if isa(destination,Integer)
#     function g(l,snd)
#       ptrs = Vector{Int32}(undef,1)
#       data = Vector{eltype(snd)}(undef,0)
#       JaggedArray(data,ptrs)
#     end
#     rcv = map_main(f,l_dest,snd;otherwise=g,main=destination)
#   else
#     @assert destination === :all
#     rcv = map(f,l_dest,snd)
#   end
#   rcv
# end

# function PartitionedArrays.gather_impl!(
#   rcv::JaggedArray,
#   snd::PTArray,
#   destination,
#   ::Type{T}) where T

#   @assert length(snd) == length(rcv)
#   for k in eachindex(snd)
#     sndk = snd[k]
#     rcvk = rcv[k]
#     gather_impl!(rcvk,sndk,destination,T)
#   end
# end

# function PartitionedArrays.allocate_scatter_impl(snd,source,::Type{T}) where T
#   similar(snd,T)
# end

# function PartitionedArrays.allocate_scatter_impl(
#   snd,
#   source,
#   ::Type{T}) where T <:AbstractVector

#   counts = map(snd) do snd
#     map(length,snd)
#   end
#   counts_scat = scatter(counts;source)
#   S = eltype(T)
#   map(counts_scat) do count
#     Vector{S}(undef,count)
#   end
# end

# function PartitionedArrays.scatter_impl!(rcv,snd,source,::Type{T}) where T
#   @assert source !== :all "Scatter all not implemented"
#   @assert length(snd[source]) == length(rcv)
#   for i in 1:length(snd)
#     rcv[i] = snd[source][i]
#   end
#   rcv
# end

# function PartitionedArrays.allocate_emit_impl(snd,source,::Type{T}) where T
#   @assert source !== :all "Scatter all not implemented"
#   similar(snd)
# end

# function PartitionedArrays.allocate_emit_impl(snd,source,::Type{T}) where T<:AbstractVector
#   @assert source !== :all "Scatter all not implemented"
#   n = map(length,snd)
#   n_all = emit(n;source)
#   S = eltype(T)
#   map(n_all) do n
#     Vector{S}(undef,n)
#   end
# end

# function PartitionedArrays.emit_impl!(rcv,snd,source,::Type{T}) where T
#   @assert source !== :all "Emit all not implemented"
#   for i in eachindex(rcv)
#     rcv[i] = snd[source]
#   end
#   rcv
# end

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
function PartitionedArrays.p_vector_cache_impl(
  ::Type{<:PTArray},vector_partition,index_partition)

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
  ptrs = local_indices_snd.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = PTArray([copy(data) for _ = eachindex(values)])
  buffer_snd = PTJaggedArray(ptdata,ptrs)
  ptrs = local_indices_rcv.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = PTArray([copy(data) for _ = eachindex(values)])
  buffer_rcv = PTJaggedArray(ptdata,ptrs)
  buffer_snd,buffer_rcv
end

function PartitionedArrays.assemble_impl!(f,vector_partition,cache,::Type{<:PTVectorAssemblyCache})
  buffer_snd = map(vector_partition,cache) do values,cache
    local_indices_snd = cache.local_indices_snd
    for (p,lid) in enumerate(local_indices_snd.data)
      for k in eachindex(values)
        cache.buffer_snd.data[k][p] = values[k][lid]
      end
    end
    cache.buffer_snd
  end
  neighbors_snd,neighbors_rcv,buffer_rcv = map(cache) do cache
    cache.neighbors_snd,cache.neighbors_rcv,cache.buffer_rcv
  end |> tuple_of_arrays
  graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
  t = exchange!(buffer_rcv,buffer_snd,graph)
  # Fill values from rcv buffer asynchronously
  @async begin
    wait(t)
    map(vector_partition,cache) do values,cache
      local_indices_rcv = cache.local_indices_rcv
      for (p,lid) in enumerate(local_indices_rcv.data)
        for k in eachindex(values)
          values[k][lid] = f(values[k][lid],cache.buffer_rcv.data[k][p])
        end
      end
    end
    nothing
  end
end

function PartitionedArrays.exchange_impl!(
  rcv,snd,graph,::Type{T}) where T<:AbstractVector{<:AbstractVector}

  @assert is_consistent(graph)
  @assert eltype(rcv) <: PTJaggedArray
  snd_ids = graph.snd
  rcv_ids = graph.rcv
  @assert length(rcv_ids) == length(rcv)
  @assert length(rcv_ids) == length(snd)
  for rcv_id in 1:length(rcv_ids)
    for (i, snd_id) in enumerate(rcv_ids[rcv_id])
      snd_snd_id = JaggedArray(snd[snd_id])
      j = first(findall(k->k==rcv_id,snd_ids[snd_id]))
      ptrs_rcv = rcv[rcv_id].ptrs
      ptrs_snd = snd_snd_id.ptrs
      @assert ptrs_rcv[i+1]-ptrs_rcv[i] == ptrs_snd[j+1]-ptrs_snd[j]
      for p in 1:(ptrs_rcv[i+1]-ptrs_rcv[i])
        p_rcv = p+ptrs_rcv[i]-1
        p_snd = p+ptrs_snd[j]-1
        for k in eachindex(snd_snd_id.data)
          rcv[rcv_id].data[k][p_rcv] = snd_snd_id.data[k][p_snd]
        end
      end
    end
  end
  @async rcv
end
