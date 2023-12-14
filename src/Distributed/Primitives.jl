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
