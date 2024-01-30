
function PartitionedArrays.nziterator(a::ParamArray{<:AbstractSparseMatrixCSC},args...)
  NZIteratorCSC(a)
end

function PartitionedArrays.nziterator(a::ParamArray{<:SparseMatrixCSR},args...)
  NZIteratorCSR(a)
end

function PartitionedArrays.nzindex(a::ParamArray{<:AbstractSparseMatrix},args...)
  nzindex(testitem(a),args...)
end

Base.first(a::NZIteratorCSC{<:ParamArray}) = first(a.matrix)
Base.length(a::NZIteratorCSC{<:ParamArray}) = length(first(a))

@inline function Base.iterate(a::NZIteratorCSC{<:ParamArray})
  if nnz(first(a)) == 0
    return nothing
  end
  col = 0
  knext = nothing
  while knext === nothing
    col += 1
    ks = nzrange(first(a),col)
    knext = iterate(ks)
  end
  k,kstate = knext
  i = Int(rowvals(first(a))[k])
  j = col
  v = map(a.matrix) do matrix
    nonzeros(matrix)[k]
  end
  (i,j,v),(col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSC{<:ParamArray},state)
  col,kstate = state
  ks = nzrange(first(a),col)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if col == size(first(a),2)
        return nothing
      end
      col += 1
      ks = nzrange(first(a),col)
      knext = iterate(ks)
    end
  end
  k,kstate = knext
  i = Int(rowvals(first(a))[k])
  j = col
  v = map(a.matrix) do matrix
    nonzeros(matrix)[k]
  end
  (i,j,v),(col,kstate)
end

Base.first(a::NZIteratorCSR{<:ParamArray}) = first(a.matrix)
Base.length(a::NZIteratorCSR{<:ParamArray}) = length(first(a))

@inline function Base.iterate(a::NZIteratorCSR{<:ParamArray})
  if nnz(first(a)) == 0
    return nothing
  end
  row = 0
  knext = nothing
  while knext === nothing
    row += 1
    ks = nzrange(first(a),row)
    knext = iterate(ks)
  end
  k, kstate = knext
  i = row
  j = Int(colvals(first(a))[k]+getoffset(first(a)))
  v = map(a.matrix) do matrix
    nonzeros(matrix)[k]
  end
  (i,j,v),(row,kstate)
end

@inline function Base.iterate(a::NZIteratorCSR{<:ParamArray},state)
  row,kstate = state
  ks = nzrange(a.matrix,row)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if row == size(a.matrix,1)
        return nothing
      end
      row += 1
      ks = nzrange(a.matrix,row)
      knext = iterate(ks)
    end
  end
  k, kstate = knext
  i = row
  j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
  v = map(a.matrix) do matrix
    nonzeros(matrix)[k]
  end
  (i,j,v),(row,kstate)
end

function PartitionedArrays.p_sparse_matrix_cache_impl(
  ::Type{<:ParamArray},
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
    rewind_ptrs!(ptrs)
    k_snd = JaggedArray(k_snd_data,ptrs)
    gi_snd = JaggedArray(gi_snd_data,ptrs)
    gj_snd = JaggedArray(gj_snd_data,ptrs)
    k_snd,gi_snd,gj_snd
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
  k_snd,gi_snd,gj_snd = map(
    setup_snd,part,parts_snd,row_partition,col_partition,matrix_partition) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  gi_rcv = exchange_fetch(gi_snd,graph)
  gj_rcv = exchange_fetch(gj_snd,graph)
  k_rcv = map(setup_rcv,part,row_partition,col_partition,gi_rcv,gj_rcv,matrix_partition)
  buffers = map(assembly_buffers,matrix_partition,k_snd,k_rcv) |> tuple_of_arrays
  cache = map(VectorAssemblyCache,parts_snd,parts_rcv,k_snd,k_rcv,buffers...)
  map(ParamSparseMatrixAssemblyCache,cache)
end

struct ParamSparseMatrixAssemblyCache
  cache::ParamVectorAssemblyCache
end

Base.reverse(a::ParamSparseMatrixAssemblyCache) = ParamSparseMatrixAssemblyCache(reverse(a.cache))
PartitionedArrays.copy_cache(a::ParamSparseMatrixAssemblyCache) = ParamSparseMatrixAssemblyCache(copy_cache(a.cache))

Base.length(a::LocalView{T,N,<:ParamArray}) where {T,N} = length(a.plids_to_value)
Base.size(a::LocalView{T,N,<:ParamArray}) where {T,N} = (length(a),)

function Base.getindex(a::LocalView{T,N,<:ParamArray},i::Integer...) where {T,N}
  LocalView(a.plids_to_value[i...],a.d_to_lid_to_plid)
end

struct ParamSubSparseMatrix{T,A,B,C}
  array::ParamArray{SubSparseMatrix{T,A,B,C},2,Vector{SubSparseMatrix{T,A,B,C}}}
end

function PartitionedArrays.SubSparseMatrix(
  parent::ParamArray{<:AbstractSparseMatrix},
  indices::Tuple,
  inv_indices::Tuple)

  array = map(a -> SubSparseMatrix(a,indices,inv_indices),parent)
  ParamSubSparseMatrix(array)
end

Base.size(a::ParamSubSparseMatrix) = size(first(a))
Base.length(a::ParamSubSparseMatrix) = length(a.array)
Base.eltype(::ParamSubSparseMatrix{T}) where T = T
Base.eltype(::Type{<:ParamSubSparseMatrix{T}}) where T = T
Base.ndims(::ParamSubSparseMatrix) = 2
Base.ndims(::Type{<:ParamSubSparseMatrix}) = 2
Base.IndexStyle(::Type{<:ParamSubSparseMatrix}) = IndexCartesian()
function Base.getindex(a::ParamSubSparseMatrix,i::Integer,j::Integer)
  map(a.array) do ai
    getindex(ai,i,j)
  end
end
Base.first(a::ParamSubSparseMatrix) = a.array[1]
function LinearAlgebra.mul!(c::ParamArray,a::ParamSubSparseMatrix,args...)
  map(c,a) do ci,ai
    mul!(ci,ai,args...)
  end
  c
end
function LinearAlgebra.fillstored!(a::ParamSubSparseMatrix,v)
  av = map(a.array) do ai
    fillstored!(ai,v)
  end
  ParamSubSparseMatrix(av)
end
