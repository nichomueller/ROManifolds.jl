function PartitionedArrays.own_values(a::ConsecutiveVectorOfVectors,indices)
  ConsecutiveArrayOfArrays(a.data[own_to_local(indices),:])
end

function PartitionedArrays.ghost_values(a::ConsecutiveVectorOfVectors,indices)
  ConsecutiveArrayOfArrays(a.data[ghost_to_local(indices),:])
end

struct ParamJaggedArray{T,L,Ti} <: AbstractVector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Ti},Base.Slice{Base.OneTo{Ti}}},false}}
  data::ConsecutiveVectorOfVectors{T,L}
  ptrs::Vector{Ti}

  function ParamJaggedArray(
    data::ConsecutiveVectorOfVectors{T,L},
    ptrs::Vector{Ti}
    ) where {T,L,Ti}

    new{T,L,Ti}(data,ptrs)
  end

  function ParamJaggedArray{T,Ti}(
    data::ConsecutiveVectorOfVectors{T,L},
    ptrs::Vector{Ti}
    ) where {T,L,Ti}

    new{T,L,Ti}(data,convert(Vector{Ti},ptrs))
  end
end

ParamDataStructures.param_length(a::ParamJaggedArray{T,L}) where {T,L} = L

PartitionedArrays.JaggedArray(data::AbstractParamVector,ptrs) = ParamJaggedArray(data,ptrs)
PartitionedArrays.JaggedArray(a::AbstractParamVector{T}) where T = ParamJaggedArray{T,Int32}(a)
PartitionedArrays.JaggedArray(a::ParamJaggedArray) = a

PartitionedArrays.JaggedArray{T,Ti}(a::ParamJaggedArray{T,Ti}) where {T,Ti} = a
function PartitionedArrays.JaggedArray{T,Ti}(
  a::AbstractArray{<:AbstractParamVector{<:Any,L}}) where {T,Ti,L}

  n = length(a)
  ptrs = Vector{Ti}(undef,n+1)
  u = one(eltype(ptrs))
  @inbounds for i in 1:n
    ai = a[i]
    ai1 = testitem(ai)
    ptrs[i+1] = length(ai1)
  end
  length_to_ptrs!(ptrs)
  ndata = ptrs[end]-u
  data = Vector{T}(undef,ndata)
  pdata = array_of_consecutive_arrays(data,L)
  p = 1
  @inbounds for i in 1:n
    ai = a[i]
    for j in 1:length(ai.data)
      aij = ai.data[j]
      pdata.data[p] = aij
      p += 1
    end
  end
  ParamJaggedArray(data,ptrs)
end

Base.size(a::ParamJaggedArray) = (length(a.ptrs)-1,param_length(a))

function Base.getindex(a::ParamJaggedArray,i::Int,j::Int)
  u = one(eltype(a.ptrs))
  pini = a.ptrs[i]
  pend = a.ptrs[i+1]-u
  view(a.data,pini:pend,j)
end

function Base.setindex!(a::ParamJaggedArray,v,i::Int,j::Int)
  u = one(eltype(a.ptrs))
  pini = a.ptrs[i]
  pend = a.ptrs[i+1]-u
  a.data[pini:pend,j] = v
end

function PartitionedArrays.assembly_buffers(
  values::AbstractParamVector{T,L},
  local_indices_snd,
  local_indices_rcv) where {T,L}

  ptrs = local_indices_snd.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = array_of_similar_arrays(data,L)
  buffer_snd = JaggedArray(ptdata,ptrs)
  ptrs = local_indices_rcv.ptrs
  data = zeros(T,ptrs[end]-1)
  ptdata = array_of_consecutive_arrays(data,L)
  buffer_rcv = JaggedArray(ptdata,ptrs)
  buffer_snd,buffer_rcv
end

struct ParamVectorAssemblyCache{T,L}
  neighbors_snd::Vector{Int32}
  neighbors_rcv::Vector{Int32}
  local_indices_snd::JaggedArray{Int32,Int32}
  local_indices_rcv::JaggedArray{Int32,Int32}
  buffer_snd::ParamJaggedArray{T,L,Int32}
  buffer_rcv::ParamJaggedArray{T,L,Int32}
end

function PartitionedArrays.VectorAssemblyCache(
  neighbors_snd,
  neighbors_rcv,
  local_indices_snd,
  local_indices_rcv,
  buffer_snd::ParamJaggedArray{T,Int32},
  buffer_rcv::ParamJaggedArray{T,Int32}) where T

  ParamVectorAssemblyCache(
    neighbors_snd,
    neighbors_rcv,
    local_indices_snd,
    local_indices_rcv,
    buffer_snd,
    buffer_rcv)
end

function Base.reverse(a::ParamVectorAssemblyCache)
  VectorAssemblyCache(
    a.neighbors_rcv,
    a.neighbors_snd,
    a.local_indices_rcv,
    a.local_indices_snd,
    a.buffer_rcv,
    a.buffer_snd)
end

function PartitionedArrays.copy_cache(a::ParamVectorAssemblyCache)
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

struct ParamJaggedArrayAssemblyCache{T}
  cache::ParamVectorAssemblyCache{T}
end

function PartitionedArrays.JaggedArrayAssemblyCache(cache::ParamVectorAssemblyCache)
  ParamJaggedArrayAssemblyCache(cache)
end

Base.reverse(a::ParamJaggedArrayAssemblyCache) = ParamJaggedArrayAssemblyCache(reverse(a.cache))

function PartitionedArrays.copy_cache(a::ParamJaggedArrayAssemblyCache)
  ParamJaggedArrayAssemblyCache(PartitionedArrays.copy_cache(a.cache))
end

function PartitionedArrays.assemble_impl!(
  f,vector_partition,cache,::Type{<:ParamJaggedArrayAssemblyCache})
  vcache = map(i->i.cache,cache)
  data = map(getdata,vector_partition)
  assemble!(f,data,vcache)
end

# new Psparse matrix code

#TODO does it all work with a simple hack to setindex of param arrays?

function PartitionedArrays.split_format_locally(A::ParamSparseMatrix{Tv,Ti,L},rows,cols) where {Tv,Ti,L}
  n_own_rows = own_length(rows)
  n_own_cols = own_length(cols)
  n_ghost_rows = ghost_length(rows)
  n_ghost_cols = ghost_length(cols)
  rows_perm = local_permutation(rows)
  cols_perm = local_permutation(cols)
  n_own_own = 0
  n_own_ghost = 0
  n_ghost_own = 0
  n_ghost_ghost = 0
  for (i,j,v) in nziterator(A)
      ip = rows_perm[i]
      jp = cols_perm[j]
      if ip <= n_own_rows && jp <= n_own_cols
          n_own_own += 1
      elseif ip <= n_own_rows
          n_own_ghost += 1
      elseif jp <= n_own_cols
          n_ghost_own += 1
      else
          n_ghost_ghost += 1
      end
  end
  own_own = (I=zeros(Ti,n_own_own),J=zeros(Ti,n_own_own),V=array_of_consecutive_arrays(zeros(Tv,n_own_own),L))
  own_ghost = (I=zeros(Ti,n_own_ghost),J=zeros(Ti,n_own_ghost),V=array_of_consecutive_arrays(zeros(Tv,n_own_ghost),L))
  ghost_own = (I=zeros(Ti,n_ghost_own),J=zeros(Ti,n_ghost_own),V=array_of_consecutive_arrays(zeros(Tv,n_ghost_own),L))
  ghost_ghost = (I=zeros(Ti,n_ghost_ghost),J=zeros(Ti,n_ghost_ghost),V=array_of_consecutive_arrays(zeros(Tv,n_ghost_ghost),L))
  n_own_own = 0
  n_own_ghost = 0
  n_ghost_own = 0
  n_ghost_ghost = 0
  for (i,j,v) in nziterator(A)
      ip = rows_perm[i]
      jp = cols_perm[j]
      if ip <= n_own_rows && jp <= n_own_cols
          n_own_own += 1
          own_own.I[n_own_own] = ip
          own_own.J[n_own_own] = jp
          own_own.V[n_own_own] = v
      elseif ip <= n_own_rows
          n_own_ghost += 1
          own_ghost.I[n_own_ghost] = ip
          own_ghost.J[n_own_ghost] = jp-n_own_cols
          own_ghost.V[n_own_ghost] = v
      elseif jp <= n_own_cols
          n_ghost_own += 1
          ghost_own.I[n_ghost_own] = ip-n_own_cols
          ghost_own.J[n_ghost_own] = jp
          ghost_own.V[n_ghost_own] = v
      else
          n_ghost_ghost += 1
          ghost_ghost.I[n_ghost_ghost] = i-n_own_rows
          ghost_ghost.J[n_ghost_ghost] = j-n_own_cols
          ghost_ghost.V[n_ghost_ghost] = v
      end
  end
  TA = typeof(A)
  A1 = compresscoo(TA,own_own...,n_own_rows  ,n_own_cols)
  A2 = compresscoo(TA,own_ghost...,n_own_rows  ,n_ghost_cols)
  A3 = compresscoo(TA,ghost_own...,n_ghost_rows,n_own_cols)
  A4 = compresscoo(TA,ghost_ghost...,n_ghost_rows,n_ghost_cols)
  blocks = split_matrix_blocks(A1,A2,A3,A4)
  B = split_matrix(blocks,rows_perm,cols_perm)
  c1 = precompute_nzindex(A1,own_own.I,own_own.J)
  c2 = precompute_nzindex(A2,own_ghost.I,own_ghost.J)
  c3 = precompute_nzindex(A3,ghost_own.I,ghost_own.J)
  c4 = precompute_nzindex(A4,ghost_ghost.I,ghost_ghost.J)
  own_own_V = own_own.V
  own_ghost_V = own_ghost.V
  ghost_own_V = ghost_own.V
  ghost_ghost_V = ghost_ghost.V
  cache = (c1,c2,c3,c4,own_own_V,own_ghost_V,ghost_own_V,ghost_ghost_V)
  B,cache
end

# function PartitionedArrays.split_format_locally!(
#   B::AbstractSplitMatrix{<:ParamSparseMatrix{Tv,Ti,L}},
#   A,rows,cols,cache) where {Tv,Ti,L}

#   (c1,c2,c3,c4,own_own_V,own_ghost_V,ghost_own_V,ghost_ghost_V) = cache
#   n_own_rows = own_length(rows)
#   n_own_cols = own_length(cols)
#   n_ghost_rows = ghost_length(rows)
#   n_ghost_cols = ghost_length(cols)
#   rows_perm = local_permutation(rows)
#   cols_perm = local_permutation(cols)
#   n_own_own = 0
#   n_own_ghost = 0
#   n_ghost_own = 0
#   n_ghost_ghost = 0
#   for (i,j,v) in nziterator(A)
#       ip = rows_perm[i]
#       jp = cols_perm[j]
#       if ip <= n_own_rows && jp <= n_own_cols
#           n_own_own += 1
#           own_own_V.data[n_own_own,:] = v.data
#       elseif ip <= n_own_rows
#           n_own_ghost += 1
#           own_ghost_V.data[n_own_ghost,:] = v.data
#       elseif jp <= n_own_cols
#           n_ghost_own += 1
#           ghost_own_V.data[n_ghost_own,:] = v.data
#       else
#           n_ghost_ghost += 1
#           ghost_ghost_V.data[n_ghost_ghost,:] = v.data
#       end
#   end
#   setcoofast!(B.blocks.own_own,own_own_V,c1)
#   setcoofast!(B.blocks.own_ghost,own_ghost_V,c2)
#   setcoofast!(B.blocks.ghost_own,ghost_own_V,c3)
#   setcoofast!(B.blocks.ghost_ghost,ghost_ghost_V,c4)
#   B
# end

function psparse_assemble_impl(
  A,
  ::Type{<:AbstractSplitMatrix{<:ParamSparseMatrix{Tv,Ti,L}}},
  rows;
  reuse=Val(false),
  assembly_neighbors_options_cols=(;)) where {Tv,Ti,L}

  function setup_cache_snd(A,parts_snd,rows_sa,cols_sa)
    A_ghost_own   = A.blocks.ghost_own
    A_ghost_ghost = A.blocks.ghost_ghost
    gen = ( owner=>i for (i,owner) in enumerate(parts_snd) )
    owner_to_p = Dict(gen)
    ptrs = zeros(Int32,length(parts_snd)+1)
    ghost_to_owner_row = ghost_to_owner(rows_sa)
    ghost_to_global_row = ghost_to_global(rows_sa)
    own_to_global_col = own_to_global(cols_sa)
    ghost_to_global_col = ghost_to_global(cols_sa)
    for (i,_,_) in nziterator(A_ghost_own)
        owner = ghost_to_owner_row[i]
        ptrs[owner_to_p[owner]+1] += 1
    end
    for (i,_,_) in nziterator(A_ghost_ghost)
        owner = ghost_to_owner_row[i]
        ptrs[owner_to_p[owner]+1] += 1
    end
    length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    I_snd_data = zeros(Int,ndata)
    J_snd_data = zeros(Int,ndata)
    V_snd_data = zeros(Tv,ndata)
    pV_snd_data = array_of_consecutive_arrays(V_snd_data,L)
    k_snd_data = zeros(Int32,ndata)
    nnz_ghost_own = 0
    for (k,(i,j,v)) in enumerate(nziterator(A_ghost_own))
        owner = ghost_to_owner_row[i]
        p = ptrs[owner_to_p[owner]]
        I_snd_data[p] = ghost_to_global_row[i]
        J_snd_data[p] = own_to_global_col[j]
        pV_snd_data[p] = v
        k_snd_data[p] = k
        ptrs[owner_to_p[owner]] += 1
        nnz_ghost_own += 1
    end
    for (k,(i,j,v)) in enumerate(nziterator(A_ghost_ghost))
        owner = ghost_to_owner_row[i]
        p = ptrs[owner_to_p[owner]]
        I_snd_data[p] = ghost_to_global_row[i]
        J_snd_data[p] = ghost_to_global_col[j]
        pV_snd_data[p] = v
        k_snd_data[p] = k+nnz_ghost_own
        ptrs[owner_to_p[owner]] += 1
    end
    rewind_ptrs!(ptrs)
    I_snd = JaggedArray(I_snd_data,ptrs)
    J_snd = JaggedArray(J_snd_data,ptrs)
    V_snd = JaggedArray(V_snd_data,ptrs)
    k_snd = JaggedArray(k_snd_data,ptrs)
    (;I_snd,J_snd,V_snd,k_snd,parts_snd)
  end
  function setup_cache_rcv(I_rcv,J_rcv,V_rcv,parts_rcv)
    k_rcv_data = zeros(Int32,length(I_rcv.data))
    k_rcv = JaggedArray(k_rcv_data,I_rcv.ptrs)
    (;I_rcv,J_rcv,V_rcv,k_rcv,parts_rcv)
  end
  function setup_touched_col_ids(A,cache_rcv,cols_sa)
    J_rcv_data = cache_rcv.J_rcv.data
    l1 = nnz(A.own_ghost)
    l2 = length(J_rcv_data)
    J_aux = zeros(Int,l1+l2)
    ghost_to_global_col = ghost_to_global(cols_sa)
    for (p,(_,j,_)) in enumerate(nziterator(A.own_ghost))
        J_own_ghost[p] = ghost_to_global_col[j]
    end
    J_aux[l1.+(1:l2)] = J_rcv_data
    J_aux
  end
  function setup_own_triplets(A,cache_rcv,rows_sa,cols_sa)
    nz_own_own = findnz(A.blocks.own_own)
    nz_own_ghost = findnz(A.blocks.own_ghost)
    I_rcv_data = cache_rcv.I_rcv.data
    J_rcv_data = cache_rcv.J_rcv.data
    V_rcv_data = cache_rcv.V_rcv.data
    k_rcv_data = cache_rcv.k_rcv.data
    global_to_own_col = global_to_own(cols_sa)
    is_ghost = findall(j->global_to_own_col[j]==0,J_rcv_data)
    is_own = findall(j->global_to_own_col[j]!=0,J_rcv_data)
    I_rcv_own = view(I_rcv_data,is_own)
    J_rcv_own = view(J_rcv_data,is_own)
    V_rcv_own = view(V_rcv_data.data,is_own,:)
    k_rcv_own = view(k_rcv_data,is_own)
    I_rcv_ghost = view(I_rcv_data,is_ghost)
    J_rcv_ghost = view(J_rcv_data,is_ghost)
    V_rcv_ghost = view(V_rcv_data.data,is_ghost,:)
    k_rcv_ghost = view(k_rcv_data,is_ghost)
    # After this col ids in own_ghost triplet remain global
    map_global_to_own!(I_rcv_own,rows_sa)
    map_global_to_own!(J_rcv_own,cols_sa)
    map_global_to_own!(I_rcv_ghost,rows_sa)
    map_ghost_to_global!(nz_own_ghost[2],cols_sa)
    own_own_I = vcat(nz_own_own[1],I_rcv_own)
    own_own_J = vcat(nz_own_own[2],J_rcv_own)
    own_own_V = ConsecutiveArrayOfArrays(vcat(nz_own_own[3].data,V_rcv_own))
    own_own_triplet = (own_own_I,own_own_J,own_own_V)
    own_ghost_I = vcat(nz_own_ghost[1],I_rcv_ghost)
    own_ghost_J = vcat(nz_own_ghost[2],J_rcv_ghost)
    own_ghost_V = ConsecutiveArrayOfArrays(vcat(nz_own_ghost[3].data,V_rcv_ghost))
    map_global_to_ghost!(nz_own_ghost[2],cols_sa)
    own_ghost_triplet = (own_ghost_I,own_ghost_J,own_ghost_V)
    triplets = (own_own_triplet,own_ghost_triplet)
    aux = (I_rcv_own,J_rcv_own,k_rcv_own,I_rcv_ghost,J_rcv_ghost,k_rcv_ghost,nz_own_own,nz_own_ghost)
    triplets,own_ghost_J,aux
  end
  function finalize_values(A,rows_fa,cols_fa,cache_snd,cache_rcv,triplets,aux)
    (own_own_triplet,own_ghost_triplet) = triplets
    (I_rcv_own,J_rcv_own,k_rcv_own,I_rcv_ghost,J_rcv_ghost,k_rcv_ghost,nz_own_own,nz_own_ghost) = aux
    map_global_to_ghost!(own_ghost_triplet[2],cols_fa)
    map_global_to_ghost!(J_rcv_ghost,cols_fa)
    TA = typeof(A.blocks.own_own)
    n_own_rows = own_length(rows_fa)
    n_own_cols = own_length(cols_fa)
    n_ghost_rows = ghost_length(rows_fa)
    n_ghost_cols = ghost_length(cols_fa)
    Ti = indextype(A.blocks.own_own)
    own_own = compresscoo(TA,own_own_triplet...,n_own_rows,n_own_cols)
    own_ghost = compresscoo(TA,own_ghost_triplet...,n_own_rows,n_ghost_cols)
    ghost_own = compresscoo(TA,Ti[],Ti[],Tv[],n_ghost_rows,n_own_cols)
    ghost_ghost = compresscoo(TA,Ti[],Ti[],Tv[],n_ghost_rows,n_ghost_cols)
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    values = split_matrix(blocks,local_permutation(rows_fa),local_permutation(rows_fa))
    nnz_own_own = nnz(own_own)
    k_own_sa = precompute_nzindex(own_own,own_own_triplet[1:2]...)
    k_ghost_sa = precompute_nzindex(own_ghost,own_ghost_triplet[1:2]...)
    for p in 1:length(I_rcv_own)
        i = I_rcv_own[p]
        j = J_rcv_own[p]
        k_rcv_own[p] = nzindex(own_own,i,j)
    end
    for p in 1:length(I_rcv_ghost)
        i = I_rcv_ghost[p]
        j = J_rcv_ghost[p]
        k_rcv_ghost[p] = nzindex(own_ghost,i,j) + nnz_own_own
    end
    cache = (;k_own_sa,k_ghost_sa,cache_snd...,cache_rcv...)
    values, cache
  end
  rows_sa = partition(axes(A,1))
  cols_sa = partition(axes(A,2))
  #rows = map(remove_ghost,rows_sa)
  cols = map(remove_ghost,cols_sa)
  parts_snd, parts_rcv = assembly_neighbors(rows_sa)
  cache_snd = map(setup_cache_snd,partition(A),parts_snd,rows_sa,cols_sa)
  I_snd = map(i->i.I_snd,cache_snd)
  J_snd = map(i->i.J_snd,cache_snd)
  V_snd = map(i->i.V_snd,cache_snd)
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t_I = exchange(I_snd,graph)
  t_J = exchange(J_snd,graph)
  t_V = exchange(V_snd,graph)
  @async begin
    I_rcv = fetch(t_I)
    J_rcv = fetch(t_J)
    V_rcv = fetch(t_V)
    cache_rcv = map(setup_cache_rcv,I_rcv,J_rcv,V_rcv,parts_rcv)
    triplets,J,aux = map(setup_own_triplets,partition(A),cache_rcv,rows_sa,cols_sa) |> tuple_of_arrays
    J_owner = find_owner(cols_sa,J)
    rows_fa = rows
    cols_fa = map(union_ghost,cols,J,J_owner)
    assembly_neighbors(cols_fa;assembly_neighbors_options_cols...)
    vals_fa, cache = map(finalize_values,partition(A),rows_fa,cols_fa,cache_snd,cache_rcv,triplets,aux) |> tuple_of_arrays
    assembled = true
    B = PSparseMatrix(vals_fa,rows_fa,cols_fa,assembled)
    if val_parameter(reuse) == false
      B
    else
      B, cache
    end
  end
end

# function PartitionedArrays.psparse_assemble_impl!(
#   B,A,
#   ::Type{<:AbstractSplitMatrix{<:ParamSparseMatrix{Tv,Ti,L}}},
#   cache) where {Tv,Ti,L}

#   function setup_snd(A,cache)
#     A_ghost_own   = A.blocks.ghost_own
#     A_ghost_ghost = A.blocks.ghost_ghost
#     nnz_ghost_own = nnz(A_ghost_own)
#     V_snd_data = cache.V_snd.data
#     k_snd_data = cache.k_snd.data
#     nz_ghost_own = nonzeros(A_ghost_own)
#     nz_ghost_ghost = nonzeros(A_ghost_ghost)
#     for p in 1:length(k_snd_data)
#       k = k_snd_data[p]
#       if k <= nnz_ghost_own
#         v = nz_ghost_own.data[k,:]
#       else
#         v = nz_ghost_ghost.data[k-nnz_ghost_own,:]
#       end
#       V_snd_data.data[p,:] = v
#     end
#   end
#   function setup_sa(B,A,cache)
#     setcoofast!(B.blocks.own_own,nonzeros(A.blocks.own_own),cache.k_own_sa)
#     setcoofast!(B.blocks.own_ghost,nonzeros(A.blocks.own_ghost),cache.k_ghost_sa)
#   end
#   function setup_rcv(B,cache)
#     B_own_own   = B.blocks.own_own
#     B_own_ghost = B.blocks.own_ghost
#     nnz_own_own = nnz(B_own_own)
#     V_rcv_data = cache.V_rcv.data
#     k_rcv_data = cache.k_rcv.data
#     nz_own_own = nonzeros(B_own_own)
#     nz_own_ghost = nonzeros(B_own_ghost)
#     for p in 1:length(k_rcv_data)
#       k = k_rcv_data[p]
#       v = V_rcv_data.data[p,:]
#       if k <= nnz_own_own
#         nz_own_own.data[k,:] += v
#       else
#         nz_own_ghost.data[k-nnz_own_own,:] += v
#       end
#     end
#   end
#   map(setup_snd,partition(A),cache)
#   parts_snd = map(i->i.parts_snd,cache)
#   parts_rcv = map(i->i.parts_rcv,cache)
#   V_snd = map(i->i.V_snd,cache)
#   V_rcv = map(i->i.V_rcv,cache)
#   graph = ExchangeGraph(parts_snd,parts_rcv)
#   t = exchange!(V_rcv,V_snd,graph)
#   map(setup_sa,partition(B),partition(A),cache)
#   @async begin
#     wait(t)
#     map(setup_rcv,partition(B),cache)
#     B
#   end
# end

function PartitionedArrays.psparse_consistent_impl(
  A,
  ::Type{<:AbstractSplitMatrix{<:ParamSparseMatrix{Tv,Ti,L}}},
  rows_co;
  reuse=Val(false)) where {Tv,Ti,L}

  function setup_snd(A,parts_snd,lids_snd,rows_co,cols_fa)
    own_to_local_row = own_to_local(rows_co)
    own_to_global_row = own_to_global(rows_co)
    own_to_global_col = own_to_global(cols_fa)
    ghost_to_global_col = ghost_to_global(cols_fa)
    li_to_p = zeros(Int32,size(A,1))
    for p in 1:length(lids_snd)
      li_to_p[lids_snd[p]] .= p
    end
    ptrs = zeros(Int32,length(parts_snd)+1)
    for (i,j,v) in nziterator(A.blocks.own_own)
      li = own_to_local_row[i]
      p = li_to_p[li]
      if p == 0
        continue
      end
      ptrs[p+1] += 1
    end
    for (i,j,v) in nziterator(A.blocks.own_ghost)
      li = own_to_local_row[i]
      p = li_to_p[li]
      if p == 0
        continue
      end
      ptrs[p+1] += 1
    end
    length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    T = eltype(A)
    I_snd = JaggedArray(zeros(Int,ndata),ptrs)
    J_snd = JaggedArray(zeros(Int,ndata),ptrs)
    V_snd = JaggedArray(array_of_consecutive_arrays(zeros(T,ndata),L),ptrs)
    k_snd = JaggedArray(zeros(Int32,ndata),ptrs)
    for (k,(i,j,v)) in enumerate(nziterator(A.blocks.own_own))
      li = own_to_local_row[i]
      p = li_to_p[li]
      if p == 0
          continue
      end
      q = ptrs[p]
      I_snd.data[q] = own_to_global_row[i]
      J_snd.data[q] = own_to_global_col[j]
      V_snd.data[q] = v
      k_snd.data[q] = k
      ptrs[p] += 1
    end
    nnz_own_own = nnz(A.blocks.own_own)
    for (k,(i,j,v)) in enumerate(nziterator(A.blocks.own_ghost))
      li = own_to_local_row[i]
      p = li_to_p[li]
      if p == 0
          continue
      end
      q = ptrs[p]
      I_snd.data[q] = own_to_global_row[i]
      J_snd.data[q] = ghost_to_global_col[j]
      V_snd.data[q] = v
      k_snd.data[q] = k+nnz_own_own
      ptrs[p] += 1
    end
    rewind_ptrs!(ptrs)
    cache_snd = (;parts_snd,lids_snd,I_snd,J_snd,V_snd,k_snd)
    cache_snd
  end
  function setup_rcv(parts_rcv,lids_rcv,I_rcv,J_rcv,V_rcv)
      cache_rcv = (;parts_rcv,lids_rcv,I_rcv,J_rcv,V_rcv)
      cache_rcv
  end
  function finalize(A,cache_snd,cache_rcv,rows_co,cols_fa)
      I_rcv_data = cache_rcv.I_rcv.data
      J_rcv_data = cache_rcv.J_rcv.data
      V_rcv_data = cache_rcv.V_rcv.data
      global_to_own_col = global_to_own(cols_fa)
      global_to_ghost_col = global_to_ghost(cols_fa)
      is_own = findall(j->global_to_own_col[j]!=0,J_rcv_data)
      is_ghost = findall(j->global_to_ghost_col[j]!=0,J_rcv_data)
      I_rcv_own = I_rcv_data[is_own]
      J_rcv_own = J_rcv_data[is_own]
      V_rcv_own = V_rcv_data[is_own]
      I_rcv_ghost = I_rcv_data[is_ghost]
      J_rcv_ghost = J_rcv_data[is_ghost]
      V_rcv_ghost = V_rcv_data[is_ghost]
      map_global_to_ghost!(I_rcv_own,rows_co)
      map_global_to_ghost!(I_rcv_ghost,rows_co)
      map_global_to_own!(J_rcv_own,cols_fa)
      map_global_to_ghost!(J_rcv_ghost,cols_fa)
      own_own = A.blocks.own_own
      own_ghost = A.blocks.own_ghost
      n_ghost_rows = ghost_length(rows_co)
      n_own_cols = own_length(cols_fa)
      n_ghost_cols = ghost_length(cols_fa)
      TA = typeof(A.blocks.ghost_own)
      ghost_own = compresscoo(TA,I_rcv_own,J_rcv_own,V_rcv_own,n_ghost_rows,n_own_cols)
      ghost_ghost = compresscoo(TA,I_rcv_ghost,J_rcv_ghost,V_rcv_ghost,n_ghost_rows,n_ghost_cols)
      K_own = precompute_nzindex(ghost_own,I_rcv_own,J_rcv_own)
      K_ghost = precompute_nzindex(ghost_ghost,I_rcv_ghost,J_rcv_ghost)
      blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
      values = split_matrix(blocks,local_permutation(rows_co),local_permutation(cols_fa))
      k_snd = cache_snd.k_snd
      V_snd = cache_snd.V_snd
      V_rcv = cache_rcv.V_rcv
      parts_snd = cache_snd.parts_snd
      parts_rcv = cache_rcv.parts_rcv
      cache = (;parts_snd,parts_rcv,k_snd,V_snd,V_rcv,is_ghost,is_own,V_rcv_own,V_rcv_ghost,K_own,K_ghost)
      values,cache
  end
  @assert matching_own_indices(axes(A,1),PRange(rows_co))
  rows_fa = partition(axes(A,1))
  cols_fa = partition(axes(A,2))
  # snd and rcv are swapped on purpose
  parts_rcv,parts_snd = assembly_neighbors(rows_co)
  lids_rcv,lids_snd = assembly_local_indices(rows_co)
  cache_snd = map(setup_snd,partition(A),parts_snd,lids_snd,rows_co,cols_fa)
  I_snd = map(i->i.I_snd,cache_snd)
  J_snd = map(i->i.J_snd,cache_snd)
  V_snd = map(i->i.V_snd,cache_snd)
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t_I = exchange(I_snd,graph)
  t_J = exchange(J_snd,graph)
  t_V = exchange(V_snd,graph)
  @async begin
      I_rcv = fetch(t_I)
      J_rcv = fetch(t_J)
      V_rcv = fetch(t_V)
      cache_rcv = map(setup_rcv,parts_rcv,lids_rcv,I_rcv,J_rcv,V_rcv)
      values,cache = map(finalize,partition(A),cache_snd,cache_rcv,rows_co,cols_fa) |> tuple_of_arrays
      B = PSparseMatrix(values,rows_co,cols_fa,A.assembled)
      if val_parameter(reuse) == false
          B
      else
          B,cache
      end
  end
end

# function psparse_consistent_impl!(
#   B,A,
#   ::Type{<:AbstractSplitMatrix{<:ParamSparseMatrix{Tv,Ti,L}}},
#   cache) where {Tv,Ti,L}

#   function setup_snd(A,cache)
#       k_snd_data = cache.k_snd.data
#       V_snd_data = cache.V_snd.data
#       nnz_own_own = nnz(A.blocks.own_own)
#       A_own_own = nonzeros(A.blocks.own_own)
#       A_own_ghost = nonzeros(A.blocks.own_ghost)
#       for (p,k) in enumerate(k_snd_data)
#           if k <= nnz_own_own
#               v = A_own_own[k]
#           else
#               v = A_own_ghost[k-nnz_own_own]
#           end
#           V_snd_data[p] = v
#       end
#   end
#   function setup_rcv(B,cache)
#       is_ghost = cache.is_ghost
#       is_own = cache.is_own
#       V_rcv_data = cache.V_rcv.data
#       K_own = cache.K_own
#       K_ghost = cache.K_ghost
#       V_rcv_own = V_rcv_data[is_own]
#       V_rcv_ghost = V_rcv_data[is_ghost]
#       setcoofast!(B.blocks.ghost_own,V_rcv_own,K_own)
#       setcoofast!(B.blocks.ghost_ghost,V_rcv_ghost,K_ghost)
#       B
#   end
#   map(setup_snd,partition(A),cache)
#   parts_snd = map(i->i.parts_snd,cache)
#   parts_rcv = map(i->i.parts_rcv,cache)
#   V_snd = map(i->i.V_snd,cache)
#   V_rcv = map(i->i.V_rcv,cache)
#   graph = ExchangeGraph(parts_snd,parts_rcv)
#   t = exchange!(V_rcv,V_snd,graph)
#   @async begin
#       wait(t)
#       map(setup_rcv,partition(B),cache)
#       B
#   end
# end

function PartitionedArrays.repartition!(
  B::PSparseMatrix{<:ParamSparseMatrix},
  A::PSparseMatrix{<:ParamSparseMatrix},
  cache)

  (V,cacheB) = cache
  function fill_values!(V,A_own_own,A_own_ghost)
    nz_own_own = nonzeros(A_own_own)
    nz_own_ghost = nonzeros(A_own_ghost)
    l1 = innerlength(nz_own_own)
    l2 = innerlength(nz_own_ghost)
    V[1:l1] = nz_own_own
    V[(1:l2).+l1] = nz_own_ghost
  end
  A_own_own = own_own_values(A)
  A_own_ghost = own_ghost_values(A)
  map(fill_values!,V,A_own_own,A_own_ghost)
  psparse!(B,V,cacheB)
end
