function Base.materialize(b::PBroadcasted{<:AbstractArray{<:PTBroadcasted}})
  own_values_out = map(Base.materialize,b.own_values)
  T = eltype(eltype(own_values_out))
  vector_partition = map(b.own_values,b.index_partition) do values,indices
    pta = parray(zeros(T,length(indices)),length(values.array))
    allocate_local_values(pta,T,indices)
  end
  a = PVector(vector_partition,b.index_partition)
  Base.materialize!(a,b)
  a
end

function Base.copy(a::PVector{<:PArray})
  values = map(local_views(a)) do v
    copy(v)
  end
  PVector(values,a.index_partition)
end

function Base.collect(v::PVector{<:PArray})
  own_values_v = own_values(v)
  own_to_global_v = map(own_to_global,partition(axes(v,1)))
  vals = gather(own_values_v,destination=:all)
  ids = gather(own_to_global_v,destination=:all)
  n = length(v)
  T = eltype(v)
  map(vals,ids) do myvals,myids
    u = Vector{T}(undef,n)
    ptu = parray(u,length(first(myvals)))
    for (a,b) in zip(myvals,myids)
      for k = eachindex(a)
        ptu[k][b] = a[k]
      end
    end
    ptu
  end |> getany
end

function PartitionedArrays.allocate_local_values(
  a::PArray,
  ::Type{T},
  indices) where T

  map(a) do ai
    similar(ai,T,local_length(indices))
  end
end

function PartitionedArrays.allocate_local_values(::Type{<:PArray},indices)
  @notimplemented "The length of the PArray is needed"
end

function PartitionedArrays.own_values(values::PArray,indices)
  map(a->own_values(a,indices),values)
end

function PartitionedArrays.ghost_values(values::PArray,indices)
  map(a->ghost_values(a,indices),values)
end

function PartitionedArrays.nziterator(a::PArray{<:AbstractSparseMatrixCSC},args...)
  NZIteratorCSC(a)
end

function PartitionedArrays.nziterator(a::PArray{<:SparseMatrixCSR},args...)
  NZIteratorCSR(a)
end

function PartitionedArrays.nzindex(a::PArray{<:AbstractSparseMatrix},args...)
  nzindex(testitem(a),args...)
end

Base.first(a::NZIteratorCSC{<:PArray}) = first(a.matrix)
Base.length(a::NZIteratorCSC{<:PArray}) = length(first(a))

@inline function Base.iterate(a::NZIteratorCSC{<:PArray})
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

@inline function Base.iterate(a::NZIteratorCSC{<:PArray},state)
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

Base.first(a::NZIteratorCSR{<:PArray}) = first(a.matrix)
Base.length(a::NZIteratorCSR{<:PArray}) = length(first(a))

@inline function Base.iterate(a::NZIteratorCSR{<:PArray})
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

@inline function Base.iterate(a::NZIteratorCSR{<:PArray},state)
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
  ::Type{<:PArray},
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
  map(PTSparseMatrixAssemblyCache,cache)
end

function PartitionedArrays.assembly_buffers(
  values::PArray,
  local_indices_snd,
  local_indices_rcv)

  T = eltype(values)
  N = length(values)
  ptrs = local_indices_snd.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = pzeros(data,N)
  buffer_snd = JaggedArray(ptdata,ptrs)
  ptrs = local_indices_rcv.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = pzeros(data,N)
  buffer_rcv = JaggedArray(ptdata,ptrs)
  buffer_snd,buffer_rcv
end

function PartitionedArrays.from_trivial_partition!(
  c::PVector{<:PArray},c_in_main::PVector{<:PArray})

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
  b::PVector{<:PArray},
  row_partition_in_main)

  destination = 1
  T = eltype(b)
  b_in_main = similar(b,T,PRange(row_partition_in_main))
  fill!(b_in_main,zero(T))
  map(
    own_values(b),
    partition(b_in_main),
    partition(axes(b,1))) do bown,my_b_in_main,indices

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
  a::PSparseMatrix{<:PArray{M}},
  row_partition_in_main=trivial_partition(partition(axes(a,1))),
  col_partition_in_main=trivial_partition(partition(axes(a,2)))) where M

  destination = 1
  Ta = eltype(a)
  I,J,V = map(
    partition(a),
    partition(axes(a,1)),
    partition(axes(a,2))) do a,row_indices,col_indices

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
    myV = pzeros(zeros(Ta,n),length(a))
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
  values = map(
    I,
    J,
    V,
    row_partition_in_main,
    col_partition_in_main) do myI,myJ,myV,row_indices,col_indices

    m = local_length(row_indices)
    n = local_length(col_indices)
    map(myV) do myV
      compresscoo(M,myI,myJ,myV,m,n)
    end
  end
  PSparseMatrix(values,row_partition_in_main,col_partition_in_main)
end

function PartitionedArrays.assemble_coo!(
  I,
  J,
  V::AbstractVector{<:PArray},
  row_partition)

  function setup_snd(part,parts_snd,row_lids,coo_values)
    global_to_local_row = global_to_local(row_lids)
    local_row_to_owner = local_to_owner(row_lids)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_gi,k_gj,k_v = coo_values
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    gi_snd_data = zeros(eltype(k_gi),ptrs[end]-1)
    gj_snd_data = zeros(eltype(k_gj),ptrs[end]-1)
    snd_data = zeros(eltype(k_v),ptrs[end]-1)
    v_snd_data = pzeros(snd_data,length(k_v))
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        gj = k_gj[k]
        p = ptrs[owner_to_i[owner]]
        gi_snd_data[p] = gi
        gj_snd_data[p] = gj
        ptrs[owner_to_i[owner]] += 1
        for i = eachindex(k_v)
          v = k_v[i][k]
          v_snd_data[i][p] = v
          k_v[i][k] = zero(v)
        end
      end
    end
    rewind_ptrs!(ptrs)
    gi_snd = JaggedArray(gi_snd_data,ptrs)
    gj_snd = JaggedArray(gj_snd_data,ptrs)
    v_snd = JaggedArray(v_snd_data,ptrs)
    gi_snd,gj_snd,v_snd
  end
  function setup_rcv!(coo_values,gi_rcv,gj_rcv,v_rcv)
    k_gi,k_gj,k_v = coo_values
    current_n = length(k_gi)
    new_n = current_n + length(gi_rcv.data)
    resize!(k_gi,new_n)
    resize!(k_gj,new_n)
    resize!(k_v,new_n)
    for p in 1:length(gi_rcv.data)
      k_gi[current_n+p] = gi_rcv.data[p]
      k_gj[current_n+p] = gj_rcv.data[p]
      for i = eachindex(k_v)
        k_v[i][current_n+p] = v_rcv.data[i][p]
      end
    end
  end
  part = linear_indices(row_partition)
  parts_snd, parts_rcv = assembly_neighbors(row_partition)
  coo_values = map(tuple,I,J,V)
  gi_snd,gj_snd,v_snd = map(
    setup_snd,part,parts_snd,row_partition,coo_values) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t1 = exchange(gi_snd,graph)
  t2 = exchange(gj_snd,graph)
  t3 = exchange(v_snd,graph)
  @async begin
    gi_rcv = fetch(t1)
    gj_rcv = fetch(t2)
    v_rcv = fetch(t3)
    map(setup_rcv!,coo_values,gi_rcv,gj_rcv,v_rcv)
    I,J,V
  end
end

struct PTJaggedArray{T,Ti} <: AbstractVector{SubArray{T,1,Vector{T},Tuple{UnitRange{Ti}},true}}
  data::PArray{T}
  ptrs::Vector{Ti}

  function PTJaggedArray(data::PArray{T},ptrs::Vector{Ti}) where {T,Ti}
    new{T,Ti}(data,ptrs)
  end
  function PTJaggedArray{T,Ti}(data::PArray{T},ptrs::Vector) where {T,Ti}
    new{T,Ti}(data,convert(Vector{Ti},ptrs))
  end
end

function PartitionedArrays.JaggedArray(data::PArray,ptrs)
  PTJaggedArray(data,ptrs)
end

PartitionedArrays.JaggedArray(a::AbstractArray{<:PArray{T}}) where T = JaggedArray{T,Int32}(a)
PartitionedArrays.JaggedArray(a::PTJaggedArray) = a
PartitionedArrays.JaggedArray{T,Ti}(a::PTJaggedArray{T,Ti}) where {T,Ti} = a

function PartitionedArrays.JaggedArray{T,Ti}(a::AbstractArray{<:PArray{T}}) where {T,Ti}
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

PartitionedArrays.jagged_array(data::PArray,ptrs::Vector) = PTJaggedArray(data,ptrs)

struct PTVectorAssemblyCache{T}
  neighbors_snd::Vector{Int32}
  neighbors_rcv::Vector{Int32}
  local_indices_snd::JaggedArray{Int32,Int32}
  local_indices_rcv::JaggedArray{Int32,Int32}
  buffer_snd::PTJaggedArray{T,Int32}
  buffer_rcv::PTJaggedArray{T,Int32}
end

function PartitionedArrays.VectorAssemblyCache(
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

Base.reverse(a::PTSparseMatrixAssemblyCache) = PTSparseMatrixAssemblyCache(reverse(a.cache))
PartitionedArrays.copy_cache(a::PTSparseMatrixAssemblyCache) = PTSparseMatrixAssemblyCache(copy_cache(a.cache))

Base.length(a::LocalView{T,N,<:PArray}) where {T,N} = length(a.plids_to_value)
Base.size(a::LocalView{T,N,<:PArray}) where {T,N} = (length(a),)

function Base.getindex(a::LocalView{T,N,<:PArray},i::Integer...) where {T,N}
  LocalView(a.plids_to_value[i...],a.d_to_lid_to_plid)
end

struct PTSubSparseMatrix{T,A,B,C}
  array::PArray{SubSparseMatrix{T,A,B,C},2,Vector{SubSparseMatrix{T,A,B,C}}}
end

function PartitionedArrays.SubSparseMatrix(
  parent::PArray{<:AbstractSparseMatrix},
  indices::Tuple,
  inv_indices::Tuple)

  array = map(a -> SubSparseMatrix(a,indices,inv_indices),parent)
  PTSubSparseMatrix(array)
end

Base.size(a::PTSubSparseMatrix) = size(first(a))
Base.length(a::PTSubSparseMatrix) = length(a.array)
Base.eltype(::PTSubSparseMatrix{T}) where T = T
Base.eltype(::Type{<:PTSubSparseMatrix{T}}) where T = T
Base.ndims(::PTSubSparseMatrix) = 2
Base.ndims(::Type{<:PTSubSparseMatrix}) = 2
Base.IndexStyle(::Type{<:PTSubSparseMatrix}) = IndexCartesian()
function Base.getindex(a::PTSubSparseMatrix,i::Integer,j::Integer)
  map(a.array) do ai
    getindex(ai,i,j)
  end
end
Base.first(a::PTSubSparseMatrix) = a.array[1]
function LinearAlgebra.mul!(c::PArray,a::PTSubSparseMatrix,args...)
  map(c,a) do ci,ai
    mul!(ci,ai,args...)
  end
  c
end
function LinearAlgebra.fillstored!(a::PTSubSparseMatrix,v)
  av = map(a.array) do ai
    fillstored!(ai,v)
  end
  PTSubSparseMatrix(av)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::TransientPFEOperator,
  μ::AbstractVector,
  t::T,
  xh::TransientDistributedCellField,
  γ::Tuple{Vararg{Real}}) where T

  _matdata_jacobians = TransientFETools.fill_jacobians(op,μ,t,xh,γ)
  matdata = GridapDistributed._vcat_distributed_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem,matdata)
  A
end
