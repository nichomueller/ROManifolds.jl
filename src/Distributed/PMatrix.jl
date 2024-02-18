struct PMatrix{V,A,B,C,D,T} <: AbstractMatrix{T}
  matrix_partition::A
  row_partition::B
  col_partition::C
  cache::D

  function PMatrix(
    matrix_partition,
    row_partition,
    col_partition=get_col_partition(matrix_partition,row_partition),
    cache=p_matrix_cache(matrix_partition,row_partition,col_partition))

    V = eltype(matrix_partition)
    T = eltype(V)
    A = typeof(matrix_partition)
    B = typeof(row_partition)
    C = typeof(col_partition)
    D = typeof(cache)
    new{V,A,B,C,D,T}(matrix_partition,row_partition,col_partition,cache)
  end
end

PartitionedArrays.partition(a::PMatrix) = a.matrix_partition
Base.axes(a::PMatrix) = (PRange(a.row_partition),PRange(a.col_partition))

GridapDistributed.local_views(a::PMatrix) = local_values(a)

function PartitionedArrays.local_values(a::PMatrix)
  partition(a)
end

function PartitionedArrays.own_values(a::PMatrix)
  map(partition(a),partition(axes(a,1)),partition(axes(a,2))) do values,indices_rows,indices_cols
    view(values,own_to_local(indices_rows),own_to_local(indices_cols))
  end
end

function PartitionedArrays.ghost_values(a::PMatrix)
  map(partition(a),partition(axes(a,1)),partition(axes(a,2))) do values,indices_rows,indices_cols
    view(values,ghost_to_local(indices_rows),ghost_to_local(indices_cols))
  end
end

function PartitionedArrays.own_ghost_values(a::PMatrix)
  map(own_ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

function PartitionedArrays.ghost_own_values(a::PMatrix)
  map(ghost_own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

Base.size(a::PMatrix) = map(length,axes(a))
Base.IndexStyle(::Type{<:PMatrix}) = IndexCartesian()

function Base.getindex(a::PMatrix,gi::Int,gj::Int)
  PartitionedArrays.scalar_indexing_action(a)
end

function Base.setindex!(a::PMatrix,v,gi::Int,gj::Int)
  PartitionedArrays.scalar_indexing_action(a)
end

function Base.show(io::IO,k::MIME"text/plain",data::PMatrix)
  T = eltype(partition(data))
  m,n = size(data)
  np = length(partition(data))
  map_main(partition(data)) do values
    println(io,"$(m)×$(n) PMatrix{$T} partitioned into $np parts")
  end
end

function get_col_partition(matrix_partition,row_partition)
  map(local_views(matrix_partition),local_views(row_partition)) do values,rows
    owner = part_id(rows)
    indices = axes(values,2)
    CommonColIndices(Int32.(owner),indices)
  end
end

struct CommonColIndices{T} <: AbstractLocalIndices
  owner::Int32
  indices::T
end

PartitionedArrays.part_id(a::CommonColIndices) = a.owner

PartitionedArrays.local_length(a::CommonColIndices) = length(a.indices)

function PartitionedArrays.own_to_global(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.own_to_owner(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.global_to_own(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.ghost_to_global(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.ghost_to_owner(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.global_to_ghost(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.own_to_local(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.ghost_to_local(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.local_to_own(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.local_to_ghost(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.global_to_local(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.local_to_global(a::CommonColIndices)
  a.indices
end

function PartitionedArrays.local_to_owner(a::CommonColIndices)
  a.indices
end

struct MatrixAssemblyCache
  cache::PartitionedArrays.VectorAssemblyCache
end

Base.reverse(a::MatrixAssemblyCache) = MatrixAssemblyCache(reverse(a.cache))
PartitionedArrays.copy_cache(a::MatrixAssemblyCache) = MatrixAssemblyCache(PartitionedArrays.copy_cache(a.cache))

function p_matrix_cache(matrix_partition,row_partition,col_partition)
  p_matrix_cache_impl(eltype(matrix_partition),matrix_partition,row_partition,col_partition)
end

function p_matrix_cache_impl(::Type,matrix_partition,row_partition,col_partition)
  function setup_snd(part,parts_snd,row_indices,col_indices,values)
    iterator = Iterators.product(axes(values)...)
    local_row_to_owner = local_to_owner(row_indices)
    local_to_global_row = local_to_global(row_indices)
    local_to_global_col = col_indices
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    for (li,lj) in iterator
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    k_snd_data = zeros(Int32,ptrs[end]-1)
    gi_snd_data = zeros(Int,ptrs[end]-1)
    gj_snd_data = zeros(Int,ptrs[end]-1)
    for (k,(li,lj)) in enumerate(iterator)
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
    k_snd, gi_snd, gj_snd
  end
  function setup_rcv(part,row_indices,col_indices,gi_rcv,gj_rcv,values)
    global_to_local_row = global_to_local(row_indices)
    global_to_local_col = col_indices
    ptrs = gi_rcv.ptrs
    k_rcv_data = zeros(Int32,ptrs[end]-1)
    for p in 1:length(gi_rcv.data)
      gi = gi_rcv.data[p]
      gj = gj_rcv.data[p]
      li = global_to_local_row[gi]
      lj = global_to_local_col[gj]
      k = li+(lj-1)*size(values,1)
      k_rcv_data[p] = k
    end
    k_rcv = JaggedArray(k_rcv_data,ptrs)
    k_rcv
  end
  part = linear_indices(row_partition)
  parts_snd,parts_rcv = assembly_neighbors(row_partition)
  k_snd,gi_snd,gj_snd = map(setup_snd,part,parts_snd,row_partition,col_partition,matrix_partition) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  gi_rcv = exchange_fetch(gi_snd,graph)
  gj_rcv = exchange_fetch(gj_snd,graph)
  k_rcv = map(setup_rcv,part,row_partition,col_partition,gi_rcv,gj_rcv,matrix_partition)
  buffers = map(assembly_buffers,matrix_partition,k_snd,k_rcv) |> tuple_of_arrays
  cache = map(VectorAssemblyCache,parts_snd,parts_rcv,k_snd,k_rcv,buffers...)
  # map(MatrixAssemblyCache,cache)
  cache
end

function PartitionedArrays.assemble!(a::PMatrix)
  assemble!(+,a)
end

function PartitionedArrays.assemble!(o,a::PMatrix)
  t = assemble!(o,partition(a),a.cache)
  @async begin
    wait(t)
    map(ghost_values(a)) do a
      fill!(a,zero(eltype(a)))
    end
    map(ghost_own_values(a)) do a
      fill!(a,zero(eltype(a)))
    end
    a
  end
end

function PMatrix{V}(::UndefInitializer,row_partition,col_partition) where V
  matrix_partition = map(row_partition,col_partition) do row_indices,col_indices
    PartitionedArrays.allocate_local_values(V,row_indices,col_indices)
  end
  PMatrix(matrix_partition,row_partition,col_partition)
end

function Base.similar(a::PMatrix,::Type{T},inds::Tuple{<:PRange,<:PRange}) where T
  rows,cols = inds
  matrix_partition = map(partition(a),partition(rows),partition(cols)) do values,row_indices,col_indices
    PartitionedArrays.allocate_local_values(values,T,row_indices,col_indices)
  end
  PMatrix(matrix_partition,partition(rows),partition(cols))
end

function Base.similar(::Type{<:PMatrix{V}},inds::Tuple{<:PRange,<:PRange}) where V
  rows,cols = inds
  matrix_partition = map(partition(rows),partition(cols)) do row_indices,col_indices
    allocate_local_values(V,row_indices,col_indices)
  end
  PMatrix(matrix_partition,partition(rows),partition(cols))
end

function Base.adjoint(a::PMatrix)
  matrix_partition = map(partition(a)) do values
    adjoint(values)
  end
  row_partition = copy(a.col_partition)
  col_partition = copy(a.row_partition)
  cache = copy(a.cache)
  PMatrix(matrix_partition,row_partition,col_partition,cache)
end

function Base.copy!(a::PMatrix,b::PMatrix)
  @assert size(a) == size(b)
  copyto!(a,b)
end

function Base.copyto!(a::PMatrix,b::PMatrix)
  if partition(axes(a,1)) === partition(axes(b,1)) && partition(axes(a,2)) === partition(axes(b,2))
    map(copy!,partition(a),partition(b))
  elseif PartitionedArrays.matching_own_indices(axes(a,1),axes(b,1)) && PartitionedArrays.matching_own_indices(axes(a,2),axes(b,2))
    map(copy!,own_values(a),own_values(b))
  else
    error("Trying to copy a PMatrix into another one with a different data layout. This case is not implemented yet. It would require communications.")
  end
  a
end

function Base.fill!(a::PMatrix,v)
  map(partition(a)) do values
    fill!(values,v)
  end
  a
end

function Base.:*(a::Number,b::PMatrix)
  matrix_partition = map(partition(b)) do values
    a*values
  end
  cache = map(PartitionedArrays.copy_cache,b.cache)
  PMatrix(matrix_partition,partition(axes(b,1)),partition(axes(b,2)),cache)
end

function Base.:*(b::PMatrix,a::Number)
  a*b
end

# Not efficient, just for convenience and debugging purposes
function Base.:*(a::PMatrix,b::PVector)
  Ta = eltype(a)
  Tb = eltype(b)
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  c = PVector{Vector{T}}(undef,partition(axes(a,1)))
  fill!(c,zero(T))
  a_in_main = to_trivial_partition(a)
  b_in_main = to_trivial_partition(b,partition(axes(a_in_main,1)))
  c_in_main = to_trivial_partition(c,partition(axes(a_in_main,2)))
  map_main(partition(c_in_main),partition(a_in_main),partition(b_in_main)) do myc,mya,myb
    myc .= mya*myb
    nothing
  end
  from_trivial_partition!(c,c_in_main)
  c
end

# Not efficient, just for convenience and debugging purposes
function Base.:*(a::PMatrix,b::PMatrix)
  Ta = eltype(a)
  Tb = eltype(b)
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  c = PMatrix{Matrix{T}}(undef,partition(axes(a,1)),partition(axes(b,2)))
  fill!(c,zero(T))
  a_in_main = to_trivial_partition(a)
  b_in_main = to_trivial_partition(b,partition(axes(a_in_main,1)))
  c_in_main = to_trivial_partition(c,partition(axes(a_in_main,2)))
  map_main(partition(c_in_main),partition(a_in_main),partition(b_in_main)) do myc,mya,myb
    myc .= mya*myb
    nothing
  end
  from_trivial_partition!(c,c_in_main)
  c
end

for op in (:+,:-)
  @eval begin
    function Base.$op(a::PMatrix)
      matrix_partition = map(partition(a)) do a
        $op(a)
      end
      cache = map(PartitionedArrays.copy_cache,a.cache)
      PMatrix(matrix_partition,partition(axes(a,1)),partition(axes(a,2)),cache)
    end
  end
end

function PartitionedArrays.to_trivial_partition(
  a::PMatrix,
  row_partition_in_main=PartitionedArrays.trivial_partition(partition(axes(a,1))),
  col_partition_in_main=PartitionedArrays.trivial_partition(partition(axes(a,2))))

  destination = 1
  T = eltype(a)
  a_in_main = similar(a,T,PRange(row_partition_in_main),PRange(col_partition_in_main))
  fill!(a_in_main,zero(T))
  map(own_values(a),partition(a_in_main),partition(axes(a,1)),partition(axes(a,2))) do aown,my_a_in_main,row_indices,col_indices
    println(own_to_global(row_indices))
    println(own_to_global(col_indices))
    println(size(my_a_in_main))
    println(size(aown))
    if part_id(row_indices) == part_id(col_indices) == destination
      my_a_in_main[own_to_global(row_indices),own_to_global(col_indices)] .= aown
    else
      my_a_in_main .= aown
    end
  end
  assemble!(a_in_main) |> wait
  a_in_main
end

# Not efficient, just for convenience and debugging purposes
function Base.:\(a::PMatrix,b::PVector)
  Ta = eltype(a)
  Tb = eltype(b)
  T = typeof(one(Ta)\one(Tb)+one(Ta)\one(Tb))
  c = PVector{Vector{T}}(undef,partition(axes(a,2)))
  fill!(c,zero(T))
  a_in_main = to_trivial_partition(a)
  b_in_main = to_trivial_partition(b,partition(axes(a_in_main,1)))
  c_in_main = to_trivial_partition(c,partition(axes(a_in_main,2)))
  map_main(partition(c_in_main),partition(a_in_main),partition(b_in_main)) do myc,mya,myb
    myc .= mya\myb
    nothing
  end
  from_trivial_partition!(c,c_in_main)
  c
end
