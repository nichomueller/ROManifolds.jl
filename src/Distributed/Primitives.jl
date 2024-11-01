function PartitionedArrays.allocate_gather_impl(
  snd,
  destination,
  ::Type{<:AbstractParamVector{T,L}}) where {T,L}

  l = map(innerlength,snd)
  l_dest = gather(l;destination)
  function f(l,snd)
    ptrs = length_to_ptrs!(pushfirst!(l,one(eltype(l))))
    ndata = ptrs[end]-1
    data = Vector{T}(undef,ndata)
    pdata = array_of_consecutive_arrays(data,L)
    ParamJaggedArray{T,Int32}(pdata,ptrs)
  end
  if isa(destination,Integer)
    function g(l,snd)
      ptrs = Vector{Int32}(undef,1)
      data = Vector{T}(undef,0)
      pdata = array_of_consecutive_arrays(data,L)
      ParamJaggedArray(pdata,ptrs)
    end
    rcv = map_main(f,l_dest,snd;otherwise=g,main=destination)
  else
    @assert destination === :all
    rcv = map(f,l_dest,snd)
  end
  rcv
end

function PartitionedArrays.allocate_scatter_impl(
  snd,
  source,
  ::Type{<:AbstractParamVector{T,L}}) where {T,L}

  counts = map(innerlength,snd)
  counts_scat = scatter(counts;source)
  map(counts_scat) do count
    data = Vector{T}(undef,count)
    array_of_consecutive_arrays(data,L)
  end
end

function PartitionedArrays.assemble_impl!(
  f,
  vector_partition,
  cache,
  ::Type{<:ParamVectorAssemblyCache})

  buffer_snd = map(vector_partition,cache) do values,cache
    local_indices_snd = cache.local_indices_snd
    for (p,lid) in enumerate(local_indices_snd.data)
      cache.buffer_snd.data[p] = values[lid]
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
        for i in param_eachindex(values)
          values.data[lid,i] = f(values.data[lid,i],cache.buffer_rcv.data.data[p,i])
        end
      end
    end
    nothing
  end
end

function PartitionedArrays.allocate_multicast_impl(
  snd,
  source,
  ::Type{<:AbstractParamVector{T,L}}) where {T,L}

  @assert source !== :all "Scatter all not implemented"
  n = map(innerlength,snd)
  n_all = multicast(n;source)
  map(n_all) do n
    data = Vector{T}(undef,n)
    pdata = array_of_consecutive_arrays(data,L)
  end
end

function PartitionedArrays.allocate_exchange_impl(
  snd,
  graph,
  ::Type{<:AbstractParamVector{T,L}}) where {T,L}

  n_snd = map(innerlength,snd)
  n_rcv = exchange_fetch(n_snd,graph)
  rcv = map(n_rcv) do n_rcv
    ptrs = zeros(Int32,length(n_rcv)+1)
    ptrs[2:end] = n_rcv
    length_to_ptrs!(ptrs)
    n_data = ptrs[end]-1
    data = Vector{T}(undef,n_data)
    pdata = array_of_consecutive_arrays(data,L)
    ParamJaggedArray(pdata,ptrs)
  end
  rcv
end
