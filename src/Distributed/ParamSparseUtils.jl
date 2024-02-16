function PartitionedArrays.nziterator(a::ParamMatrix{T,A}) where {T,A}
  if eltype(A) <: SparseMatrixCSR
    PartitionedArrays.NZIteratorCSR(a)
  else
    PartitionedArrays.NZIteratorCSC(a)
  end
end

function PartitionedArrays.nzindex(a::ParamArray,args...)
  PartitionedArrays.nzindex(first(a),args...)
end

function PartitionedArrays.nziterator(a::RB.BasicNnzSnapshots)
  PartitionedArrays.nziterator(first(get_values(a)))
end

function PartitionedArrays.nzindex(a::RB.BasicNnzSnapshots,args...)
  PartitionedArrays.nzindex(first(get_values(a)),args...)
end

function PartitionedArrays.compresscoo(
  ::Type{ParamMatrix{T,A,L}},
  I::AbstractVector,
  J::AbstractVector,
  V::ParamArray,
  args...) where {T,A,L}

  elA = eltype(A)
  v = map(V) do V
    compresscoo(elA,I,J,V,args...)
  end
  ParamArray(v)
end

@inline function Base.iterate(a::NZIteratorCSC{<:ParamMatrix})
  if nnz(a.matrix) == 0
      return nothing
  end
  col = 0
  knext = nothing
  while knext === nothing
    col += 1
    ks = nzrange(a.matrix,col)
    knext = iterate(ks)
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = map(a.matrix) do matrix
    nonzeros(matrix)[k]
  end
  pv = ParamArray(v)
  (i,j,pv),(col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSC{<:ParamMatrix},state)
  col, kstate = state
  ks = nzrange(a.matrix,col)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if col == size(a.matrix,2)
          return nothing
      end
      col += 1
      ks = nzrange(a.matrix,col)
      knext = iterate(ks)
    end
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = map(a.matrix) do matrix
    nonzeros(matrix)[k]
  end
  pv = ParamArray(v)
  (i,j,pv),(col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSR{<:ParamMatrix})
  if nnz(a.matrix) == 0
    return nothing
  end
  row = 0
  ptrs = a.matrix.rowptr
  knext = nothing
  while knext === nothing
    row += 1
    ks = nzrange(a.matrix,row)
    knext = iterate(ks)
  end
  k, kstate = knext
  i = row
  j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
  v = map(a.matrix) do matrix
    nonzeros(matrix)[k]
  end
  pv = ParamArray(v)
  (i,j,pv),(row,kstate)
end

@inline function Base.iterate(a::NZIteratorCSR{<:ParamMatrix},state)
  row, kstate = state
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
  pv = ParamArray(v)
  (i,j,pv),(row,kstate)
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

struct ParamSparseMatrixAssemblyCache{T}
  cache::ParamVectorAssemblyCache{T}
end

Base.reverse(a::ParamSparseMatrixAssemblyCache) = ParamSparseMatrixAssemblyCache(reverse(a.cache))
PartitionedArrays.copy_cache(a::ParamSparseMatrixAssemblyCache) = ParamSparseMatrixAssemblyCache(copy_cache(a.cache))

Base.length(a::LocalView{T,N,<:ParamArray}) where {T,N} = length(a.plids_to_value)
Base.size(a::LocalView{T,N,<:ParamArray}) where {T,N} = (length(a),)

function Base.getindex(a::LocalView{T,N,<:ParamArray},i::Integer...) where {T,N}
  LocalView(a.plids_to_value[i...],a.d_to_lid_to_plid)
end

struct ParamSubSparseMatrix{T,A,B,C} <: AbstractParamContainer{T,2}
  parent::A
  indices::B
  inv_indices::C
  function ParamSubSparseMatrix(
    parent::ParamMatrix{T},
    indices::Tuple,
    inv_indices::Tuple) where T

    A = typeof(parent)
    B = typeof(indices)
    C = typeof(inv_indices)
    new{T,A,B,C}(parent,indices,inv_indices)
  end
end

function PartitionedArrays.SubSparseMatrix(
  parent::ParamMatrix,
  indices::Tuple,
  inv_indices::Tuple)

  ParamSubSparseMatrix(parent,indices,inv_indices)
end

Base.size(a::ParamSubSparseMatrix) = map(length,a.indices)
Base.length(a::ParamSubSparseMatrix) = length(a.parent)
Base.eachindex(a::ParamSubSparseMatrix) = Base.OneTo(length(a))

function Base.getindex(a::ParamSubSparseMatrix,index::Int)
  PartitionedArrays.SubSparseMatrix(a.parent[index],a.indices,a.inv_indices)
end

function LinearAlgebra.mul!(
  c::ParamVector,
  a::ParamSubSparseMatrix,
  b::ParamVector,
  α::Number,
  β::Number)

  @inbounds for k = eachindex(a)
    mul!(c[k],a[k],b[k],α,β)
  end
end

function LinearAlgebra.fillstored!(a::ParamSubSparseMatrix,v)
  @inbounds for k = eachindex(a)
    LinearAlgebra.fillstored!(a[k],v)
  end
end

function PartitionedArrays.from_trivial_partition!(
  c::PVector{<:ParamArray},
  c_in_main::PVector{<:ParamArray})

  destination = 1
  consistent!(c_in_main) |> wait
  map(own_values(c),partition(c_in_main),partition(axes(c,1))) do cown,my_c_in_main,indices
    part = part_id(indices)
    map(cown,my_c_in_main) do cown,my_c_in_main
      if part == destination
        cown .= view(my_c_in_main,own_to_global(indices))
      else
        cown .= my_c_in_main
      end
    end
  end
  c
end

function PartitionedArrays.to_trivial_partition(
  b::PVector{<:ParamArray},
  row_partition_in_main)

  destination = 1
  T = eltype(b)
  b_in_main = similar(b,T,PRange(row_partition_in_main))
  fill!(b_in_main,zero(T))
  map(own_values(b),partition(b_in_main),partition(axes(b,1))) do bown,my_b_in_main,indices
    part = part_id(indices)
    map(my_b_in_main,bown) do my_b_in_main,bown
      if part == destination
        my_b_in_main[own_to_global(indices)] .= bown
      else
        my_b_in_main .= bown
      end
    end
  end
  assemble!(b_in_main) |> wait
  b_in_main
end

function PartitionedArrays.to_trivial_partition(
  a::PSparseMatrix{M},
  row_partition_in_main=trivial_partition(partition(axes(a,1))),
  col_partition_in_main=trivial_partition(partition(axes(a,2)))) where M<:ParamArray

  destination = 1
  Ta = eltype(a)
  I,J,V = map(partition(a),partition(axes(a,1)),partition(axes(a,2))) do a,row_indices,col_indices
    n = 0
    local_row_to_owner = local_to_owner(row_indices)
    owner = part_id(row_indices)
    local_to_global_row = local_to_global(row_indices)
    local_to_global_col = local_to_global(col_indices)
    for (i,j,v) in nziterator(a)
      if local_row_to_owner[i] == owner
        n += 1
      end
    end
    myI = zeros(Int,n)
    myJ = zeros(Int,n)
    myV = zero_param_array(zeros(Ta,n),length(a))
    n = 0
    for (i,j,v) in nziterator(a)
      if local_row_to_owner[i] == owner
        n += 1
        myI[n] = local_to_global_row[i]
        myJ[n] = local_to_global_col[j]
        for k in eachindex(a)
          myV[k][n] = v[k]
        end
      end
    end
    myI,myJ,myV
  end |> tuple_of_arrays
  assemble_coo!(I,J,V,row_partition_in_main) |> wait
  I,J,V = map(partition(axes(a,1)),I,J,V) do row_indices,myI,myJ,myV
    owner = part_id(row_indices)
    if owner == destination
      myI,myJ,myV
    else
      similar(myI,eltype(myI),0),similar(myJ,eltype(myJ),0),similar(myV,eltype(myV),0)
    end
  end |> tuple_of_arrays
  values = map(I,J,V,row_partition_in_main,col_partition_in_main) do myI,myJ,myV,row_indices,col_indices
    m = local_length(row_indices)
    n = local_length(col_indices)
    compresscoo(M,myI,myJ,myV,m,n)
  end
  PSparseMatrix(values,row_partition_in_main,col_partition_in_main)
end

function Base.:\(
  a::PSparseMatrix{<:ParamMatrix{Ta,A,L}},
  b::PVector{<:ParamVector{Tb,B,L}}
  ) where {Ta,Tb,A,B,L}

  T = typeof(one(Ta)\one(Tb)+one(Ta)\one(Tb))
  PT = typeof(ParamVector{Vector{T}}(undef,L))
  c = PVector{PT}(undef,partition(axes(a,2)))
  fill!(c,zero(T))
  a_in_main = to_trivial_partition(a)
  b_in_main = to_trivial_partition(b,partition(axes(a_in_main,1)))
  c_in_main = to_trivial_partition(c,partition(axes(a_in_main,2)))
  map_main(partition(c_in_main),partition(a_in_main),partition(b_in_main)) do myc, mya, myb
    myc .= mya\myb
    nothing
  end
  PartitionedArrays.from_trivial_partition!(c,c_in_main)
  c
end

function Base.:*(
  a::PSparseMatrix{<:ParamMatrix{Ta,A,L}},
  b::PVector{<:ParamVector{Tb,B,L}}
  ) where {Ta,Tb,A,B,L}

  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  PT = typeof(ParamVector{Vector{T}}(undef,L))
  c = PVector{PT}(undef,partition(axes(a,1)))
  mul!(c,a,b)
  c
end
