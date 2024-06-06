function PartitionedArrays.allocate_gather_impl(
  snd::AbstractVector{ParamArray{T,L,A}},
  destination,
  ::Type{T}) where {T,L,A}

  l = map(snd) do snd
    length(first(snd))
  end
  l_dest = gather(l;destination)
  function f(l,snd)
    ptrs = length_to_ptrs!(pushfirst!(l,one(eltype(l))))
    ndata = ptrs[end]-1
    data = Vector{T}(undef,ndata)
    ptdata = array_of_similar_arrays(data,L)
    ParamJaggedArray{T,Int32}(ptdata,ptrs)
  end
  if isa(destination,Integer)
    function g(l,snd)
      ptrs = Vector{Int32}(undef,1)
      data = Vector{T}(undef,0)
      ptdata = array_of_similar_arrays(data,L)
      ParamJaggedArray(ptdata,ptrs)
    end
    rcv = map_main(f,l_dest,snd;otherwise=g,main=destination)
  else
    @assert destination === :all
    rcv = map(f,l_dest,snd)
  end
  rcv
end

function PartitionedArrays.gather_impl!(
  rcv::AbstractVector{<:ParamJaggedArray},
  snd::AbstractVector{<:ParamArray},
  destination,
  ::Type{T}) where T

  if isa(destination,Integer)
    @assert length(rcv[destination]) == length(snd)
    for i in 1:length(snd)
      rcv[destination][i] .= snd[i]
    end
  else
    @assert destination === :all
    for j in eachindex(rcv)
      for i in 1:length(snd)
        rcv[j][i] .= snd[i]
      end
    end
  end
  rcv
end

function PartitionedArrays.allocate_scatter_impl(
  snd::AbstractVector{ParamArray{T,L,A}},
  source,
  ::Type{T}) where {T,L,A}

  counts = map(snd) do snd
    map(length,first(snd))
  end
  counts_scat = scatter(counts;source)
  map(counts_scat) do count
    data = Vector{T}(undef,count)
    array_of_similar_arrays(data,L)
  end
end

function PartitionedArrays.scatter_impl!(
  rcv::AbstractVector{<:ParamArray},
  snd::AbstractVector{<:ParamArray},
  source,
  ::Type)

  @assert source !== :all "Scatter all not implemented"
  @assert length(snd[source]) == length(rcv)
  for i in 1:length(snd)
    rcv[i] .= snd[source][i]
  end
  rcv
end

function PartitionedArrays.allocate_emit_impl(
  snd::AbstractVector{ParamArray{T,L,A}},
  source,
  ::Type) where {T,L,A}

  @assert source !== :all "Scatter all not implemented"
  n = map(snd) do snd
    length(first(snd))
  end
  n_all = emit(n;source)
  map(n_all) do n
    data = Vector{T}(undef,n)
    array_of_similar_arrays(data,L)
  end
end

function PartitionedArrays.emit_impl!(
  rcv::AbstractVector{<:ParamArray},
  snd::AbstractVector{<:ParamArray},
  source,
  ::Type)

  @assert source !== :all "Emit all not implemented"
  for i in eachindex(rcv)
    rcv[i] .= snd[source]
  end
  rcv
end

function PartitionedArrays.assemble_impl!(
  f,
  vector_partition,
  cache,
  ::Type{<:ParamVectorAssemblyCache})

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

function PartitionedArrays.assemble_impl!(
  f,
  matrix_partition,
  cache,
  ::Type{<:ParamSparseMatrixAssemblyCache})

  vcache = map(i->i.cache,cache)
  data = map(nonzeros,matrix_partition)
  assemble!(f,data,vcache)
end

function PartitionedArrays.allocate_exchange_impl(
  snd::AbstractVector{ParamJaggedArray{T,Ti,A}},
  graph,
  ::Type) where {T,Ti,A}

  n_snd = map(snd) do snd
    map(length,first(snd))
  end
  n_rcv = exchange_fetch(n_snd,graph)
  S = eltype(eltype(eltype(eltype(snd))))
  rcv = map(n_rcv) do n_rcv
    ptrs = zeros(Int32,length(n_rcv)+1)
    ptrs[2:end] = n_rcv
    length_to_ptrs!(ptrs)
    n_data = ptrs[end]-1
    data = Vector{S}(undef,n_data)
    ptdata = array_of_similar_arrays(data,length(A))
    JaggedArray(ptdata,ptrs)
  end
  rcv
end

function PartitionedArrays.exchange_impl!(
  rcv::AbstractVector{<:ParamJaggedArray},
  snd::AbstractVector{<:ParamJaggedArray},
  graph,
  ::Type)

  @assert is_consistent(graph)
  snd_ids = graph.snd
  rcv_ids = graph.rcv
  @assert length(rcv_ids) == length(rcv)
  @assert length(rcv_ids) == length(snd)
  for rcv_id in 1:length(rcv_ids)
    for (i,snd_id) in enumerate(rcv_ids[rcv_id])
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
