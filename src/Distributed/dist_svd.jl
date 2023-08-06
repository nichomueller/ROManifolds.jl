struct PMatrix{V,A,B,C,D,T} <: AbstractMatrix{T}
  values_partition::A
  row_partition::B
  col_partition::C
  cache::D

  function PMatrix(vec_of_pvecs::LazyArray{G,L,N,F} where {G,N,F}) where {L<:PVector}
    vector_partition = partition(first(vec_of_pvecs))
    row_partition = first(vec_of_pvecs).index_partition
    col_partition = nothing
    cache = first(vec_of_pvecs).cache

    msg = """ Cannot merge the partitioned vectors: they belong to different
    partitions.
    """
    @check all(map(x->getproperty(x,:index_partition),vec_of_pvecs) .== row_partition) msg
    @check all(map(x->getproperty(x,:cache)) .== cache,vec_of_pvecs) msg

    T = eltype(eltype(vector_partition))
    V = eltype(vector_partition)
    A = typeof(vector_partition)
    B = typeof(row_partition)
    C = typeof(col_partition)
    D = typeof(cache)
    new{V,A,B,C,D,T}(vector_partition,row_partition,col_partition,cache)
  end

  function PMatrix(mat_of_pvecs::LazyArray{G,L,N,F} where {G,N,F}) where {L<:PSparseMatrix}
    matrix_partition = partition(first(mat_of_pvecs))
    row_partition = first(mat_of_pvecs).row_partition
    col_partition = first(mat_of_pvecs).col_partition
    cache = first(mat_of_pvecs).cache

    msg = """ Cannot merge the nonzero vectors belonging to the partitioned
    sparse matrices: they belong to different partitions.
    """
    @check all(map(x->getproperty(x,:row_partition),mat_of_pvecs) .== row_partition) msg
    @check all(map(x->getproperty(x,:col_partition),mat_of_pvecs) .== row_partition) msg
    @check all(map(x->getproperty(x,:cache),mat_of_pvecs) .== cache) msg

    T = eltype(eltype(matrix_partition))
    V = eltype(matrix_partition)
    A = typeof(matrix_partition)
    B = typeof(row_partition)
    C = typeof(col_partition)
    D = typeof(cache)
    new{V,A,B,C,D,T}(matrix_partition,row_partition,col_partition,cache)
  end
end



function dhcat(
  pvec::L,
  vec_of_pvecs::LazyArray{G,L,N,F} where {G,N,F}) where {L<:PVector}


end

v = res_Dvec[1]

hred(x,y) = map(hcat,local_views(x),local_views(y))
N = length(res_Dvec)
ye
for n = 1:N-1
  map(hcat,local_views(res_Dvec[n]),local_views(res_Dvec[n+1]))
end
ye = map(hcat,local_views(v),local_views(v))

function rec_fun()
  N = length(res_Dvec)
  n = 1
  rcv = local_views(res_Dvec[1])
  for n = 2:N-1
    snd = local_views(res_Dvec[n])
    rcv = map(hcat,rcv,snd)
  end
  rcv
end

v = res_Dvec[1]

snd = local_views(v)
ye = map(snd) do sndi
  gather(sndi)
end

np = 4
ranks = LinearIndices((np,))
neigs_snd = map(ranks) do rank
  [1]
end
graph = ExchangeGraph(neigs_snd)

# data_rcv = exchange(vl,graph) |> fetch

n_snd = map(snd) do snd
  map(length,snd)
end
n_rcv = PartitionedArrays.exchange_fetch(n_snd,graph)
S = eltype(eltype(eltype(snd)))
rcv = map(n_rcv) do n_rcv
  ptrs = zeros(Int32,length(n_rcv)+1)
  ptrs[2:end] = n_rcv
  length_to_ptrs!(ptrs)
  n_data = ptrs[end]-1
  data = Vector{S}(undef,n_data)
  JaggedArray(data,ptrs)
end
rcv
