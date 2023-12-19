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

  function setup_snd(part,parts_snd,row_indices,col_indices,values)
    local_row_to_owner = local_to_owner(row_indices)
    local_to_global_row = local_to_global(row_indices)
    local_to_global_col = local_to_global(col_indices)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    for (li,lj,v) in nziterator(values)
      owner = local_row_to_owner[li]
      if owner != part
          ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    k_snd_data = zeros(Int32,ptrs[end]-1)
    gi_snd_data = zeros(Int,ptrs[end]-1)
    gj_snd_data = zeros(Int,ptrs[end]-1)
    for (k,(li,lj,v)) in enumerate(nziterator(values))
      owner = local_row_to_owner[li]
      if owner != part
          p = ptrs[owner_to_i[owner]]
          k_snd_data[p] = k
          gi_snd_data[p] = local_to_global_row[li]
          gj_snd_data[p] = local_to_global_col[lj]
          ptrs[owner_to_i[owner]] += 1
      end
    end
    PartitionedArrays.rewind_ptrs!(ptrs)
    k_snd = JaggedArray(k_snd_data,ptrs)
    gi_snd = JaggedArray(gi_snd_data,ptrs)
    gj_snd = JaggedArray(gj_snd_data,ptrs)
    k_snd, gi_snd, gj_snd
  end
  function setup_rcv(part,row_indices,col_indices,gi_rcv,gj_rcv,values)
    global_to_local_row = global_to_local(row_indices)
    global_to_local_col = global_to_local(col_indices)
    ptrs = gi_rcv.ptrs
    k_rcv_data = zeros(Int32,ptrs[end]-1)
    for p in 1:length(gi_rcv.data)
      gi = gi_rcv.data[p]
      gj = gj_rcv.data[p]
      li = global_to_local_row[gi]
      lj = global_to_local_col[gj]
      k = nzindex(values,li,lj)
      @boundscheck @assert k > 0 "The sparsity pattern of the ghost layer is inconsistent"
      k_rcv_data[p] = k
    end
    k_rcv = JaggedArray(k_rcv_data,ptrs)
    k_rcv
  end
  part = linear_indices(row_partition)
  parts_snd, parts_rcv = assembly_neighbors(row_partition)
  matrix_partition1 = map(matrix_partition) do matrix_partition
    first(matrix_partition)
  end
  k_snd, gi_snd, gj_snd = map(
    setup_snd,part,parts_snd,row_partition,col_partition,matrix_partition1) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  gi_rcv = exchange_fetch(gi_snd,graph)
  gj_rcv = exchange_fetch(gj_snd,graph)
  k_rcv = map(setup_rcv,part,row_partition,col_partition,gi_rcv,gj_rcv,matrix_partition1)
  buffers = map(assembly_buffers,matrix_partition,k_snd,k_rcv) |> tuple_of_arrays
  cache = map(VectorAssemblyCache,parts_snd,parts_rcv,k_snd,k_rcv,buffers...)
  map(SparseMatrixAssemblyCache,cache)
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

function VectorAssemblyCache(
  neighbors_snd,
  neighbors_rcv,
  local_indices_snd,
  local_indices_rcv,
  buffer_snd::PTJaggedArray{T,Int32},
  buffer_rcv::PTJaggedArray{T,Int32}) where T

  PTVectorAssemblyCache(
    neighbors_snd,
    neighbors_rcv,
    local_indices_snd,
    local_indices_rcv,
    buffer_snd,
    buffer_rcv)
end

function Base.reverse(a::PTVectorAssemblyCache)
  VectorAssemblyCache(
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
  VectorAssemblyCache(
    a.neighbors_snd,
    a.neighbors_rcv,
    a.local_indices_snd,
    a.local_indices_rcv,
    buffer_snd,
    buffer_rcv)
end

struct JaggedPTArrayAssemblyCache{T}
  cache::PTVectorAssemblyCache{T}
end

function JaggedArrayAssemblyCache(cache::PTVectorAssemblyCache{T}) where T
  JaggedPTArrayAssemblyCache(cache)
end

struct PTSparseMatrixAssemblyCache
  cache::PTVectorAssemblyCache
end

function SparseMatrixAssemblyCache(cache::PTVectorAssemblyCache)
  PTSparseMatrixAssemblyCache(cache)
end

Base.reverse(a::PTSparseMatrixAssemblyCache) = SparseMatrixAssemblyCache(reverse(a.cache))
PartitionedArrays.copy_cache(a::PTSparseMatrixAssemblyCache) = SparseMatrixAssemblyCache(copy_cache(a.cache))

Base.length(a::LocalView{T,N,<:PTArray}) where {T,N} = length(a.plids_to_value)
Base.size(a::LocalView{T,N,<:PTArray}) where {T,N} = (length(a),)

function Base.getindex(a::LocalView{T,N,<:PTArray},i::Integer...) where {T,N}
  LocalView(a.plids_to_value[i...],a.d_to_lid_to_plid)
end
