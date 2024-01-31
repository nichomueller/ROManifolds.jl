function GridapDistributed.to_local_indices!(
  I::AbstractVector{<:AbstractParamContainer},
  ids::PRange;
  kwargs...)

  map(I,partition(ids)) do I,indices
    lids = map(I) do i
      PartitionedArrays.to_local!(i,indices)
    end
    ParamContainer(lids)
  end
end
function GridapDistributed.to_global_indices!(
  I::AbstractVector{<:AbstractParamContainer},
  ids::PRange;
  kwargs...)

  map(I,partition(ids)) do I,indices
    gids = map(I) do i
      PartitionedArrays.to_global!(i,indices)
    end
    ParamContainer(gids)
  end
end

function GridapDistributed.get_gid_owners(
  I::AbstractVector{<:AbstractParamContainer},
  ids::PRange;
  kwargs...)

  map(I,partition(ids)) do I,indices
    gid_owners = map(I) do i
      glo_to_loc = global_to_local(indices)
      loc_to_own = local_to_owner(indices)
      map(x->loc_to_own[glo_to_loc[x]],i)
    end
    ParamContainer(gid_owners)
  end
end

function GridapDistributed.change_axes(a::ParamContainer{<:Algebra.CounterCOO},axes)
  b = map(x -> change_axes(x,axes),a)
  ParamContainer(b)
end

function GridapDistributed.change_axes(a::ParamContainer{<:Algebra.AllocationCOO},axes)
  b = map(x -> change_axes(x,axes),a)
  ParamContainer(b)
end

function Algebra.nz_counter(
  builder::PSparseMatrixBuilderCOO{ParamMatrix{T,A,L}},
  axs::Tuple{<:PRange,<:PRange}) where {T,A,L}

  test_dofs_gids_prange, trial_dofs_gids_prange = axs
  counters = map(partition(test_dofs_gids_prange),partition(trial_dofs_gids_prange)) do r,c
    axs = (Base.OneTo(local_length(r)),Base.OneTo(local_length(c)))
    counter = map(1:L) do i
      Algebra.CounterCOO{eltype(A)}(axs)
    end
    ParamContainer(counter)
  end
  DistributedParamCounterCOO(builder.par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
end

struct DistributedParamCounterCOO{A,B,C,D} <: GridapType
  par_strategy::A
  counters::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedParamCounterCOO(
    par_strategy,
    counters::AbstractArray{<:AbstractArray{<:Algebra.CounterCOO}},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(counters)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function GridapDistributed.DistributedCounterCOO(
  par_strategy,
  counters::AbstractArray{<:AbstractArray{<:Algebra.CounterCOO}},
  test_dofs_gids_prange::PRange,
  trial_dofs_gids_prange::PRange)
  DistributedParamCounterCOO(par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
end

function GridapDistributed.local_views(a::DistributedParamCounterCOO)
  a.counters
end

function GridapDistributed.local_views(
  a::DistributedParamCounterCOO,test_dofs_gids_prange,trial_dofs_gids_prange)
  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  a.counters
end

function Algebra.nz_allocation(a::DistributedParamCounterCOO)
  allocs = map(nz_allocation,a.counters)
  DistributedParamAllocationCOO(a.par_strategy,allocs,a.test_dofs_gids_prange,a.trial_dofs_gids_prange)
end

struct DistributedParamAllocationCOO{A,B,C,D} <: GridapType
  par_strategy::A
  allocs::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedParamAllocationCOO(
    par_strategy,
    allocs::AbstractArray{<:AbstractArray{<:Algebra.AllocationCOO}},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(allocs)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,allocs,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function GridapDistributed.DistributedAllocationCOO(
  par_strategy,
  allocs::AbstractArray{<:AbstractArray{<:Algebra.AllocationCOO}},
  test_dofs_gids_prange::PRange,
  trial_dofs_gids_prange::PRange)

  DistributedParamAllocationCOO(par_strategy,allocs,test_dofs_gids_prange,trial_dofs_gids_prange)
end

function GridapDistributed.change_axes(
  a::DistributedParamAllocationCOO{A,B,<:PRange,<:PRange},
  axes::Tuple{<:PRange,<:PRange}) where {A,B}

  local_axes = map(partition(axes[1]),partition(axes[2])) do rows,cols
    (Base.OneTo(local_length(rows)), Base.OneTo(local_length(cols)))
  end
  allocs = map(change_axes,a.allocs,local_axes)
  DistributedParamAllocationCOO(a.par_strategy,allocs,axes[1],axes[2])
end

function GridapDistributed.change_axes(
  a::MatrixBlock{<:DistributedParamAllocationCOO},
  axes::Tuple{<:Vector,<:Vector})

  block_ids  = CartesianIndices(a.array)
  rows,cols = axes
  array = map(block_ids) do I
    change_axes(a[I],(rows[I[1]],cols[I[2]]))
  end
  return ArrayBlock(array,a.touched)
end

function GridapDistributed.local_views(a::DistributedParamAllocationCOO)
  a.allocs
end

function GridapDistributed.local_views(
  a::DistributedParamAllocationCOO,test_dofs_gids_prange,trial_dofs_gids_prange)

  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  a.allocs
end

function GridapDistributed.local_views(a::MatrixBlock{<:DistributedParamAllocationCOO})
  array = map(local_views,a.array) |> to_parray_of_arrays
  return map(ai -> ArrayBlock(ai,a.touched),array)
end

function GridapDistributed.get_allocations(a::DistributedParamAllocationCOO)
  I,J,V = map(local_views(a)) do alloc
    i,j,v = map(alloc) do a
      a.I,a.J,a.V
    end |> tuple_of_arrays
    ParamContainer(i),ParamContainer(j),ParamContainer(v)
  end |> tuple_of_arrays
  return I,J,V
end

function GridapDistributed.get_allocations(a::ArrayBlock{<:DistributedParamAllocationCOO})
  tuple_of_array_of_parrays = map(get_allocations,a.array) |> tuple_of_arrays
  return tuple_of_array_of_parrays
end

GridapDistributed.get_test_gids(a::DistributedParamAllocationCOO)  = a.test_dofs_gids_prange
GridapDistributed.get_trial_gids(a::DistributedParamAllocationCOO) = a.trial_dofs_gids_prange
GridapDistributed.get_test_gids(a::ArrayBlock{<:DistributedParamAllocationCOO})  = map(get_test_gids,diag(a.array))
GridapDistributed.get_trial_gids(a::ArrayBlock{<:DistributedParamAllocationCOO}) = map(get_trial_gids,diag(a.array))

function Algebra.create_from_nz(a::DistributedParamAllocationCOO{<:FullyAssembledRows})
  f(x) = nothing
  A, = GridapDistributed._fa_create_from_nz_with_callback(f,a)
  return A
end

function Algebra.create_from_nz(a::ArrayBlock{<:DistributedParamAllocationCOO{<:FullyAssembledRows}})
  f(x) = nothing
  A, = GridapDistributed._fa_create_from_nz_with_callback(f,a)
  return A
end

function Algebra.create_from_nz(a::DistributedParamAllocationCOO{<:SubAssembledRows})
  f(x) = nothing
  A, = GridapDistributed._sa_create_from_nz_with_callback(f,f,a)
  return A
end

function Algebra.create_from_nz(a::ArrayBlock{<:DistributedParamAllocationCOO{<:SubAssembledRows}})
  f(x) = nothing
  A, = GridapDistributed._sa_create_from_nz_with_callback(f,f,a)
  return A
end

######vector

function Algebra.nz_allocation(
  a::DistributedParamCounterCOO{<:SubAssembledRows},
  b::PVectorCounter{<:SubAssembledRows})

  A      = nz_allocation(a)
  dofs   = b.test_dofs_gids_prange
  values = map(nz_allocation,b.counters)
  B = PVectorAllocationTrackOnlyValues(b.par_strategy,values,dofs)
  return A,B
end

function Algebra.create_from_nz(
  a::DistributedParamAllocationCOO{<:FullyAssembledRows},
  c_fespace::PVectorAllocationTrackOnlyValues{<:FullyAssembledRows})

  function callback(rows)
    GridapDistributed._rhs_callback(c_fespace,rows)
  end
  A,b = GridapDistributed._fa_create_from_nz_with_callback(callback,a)
  return A,b
end

function PartitionedArrays.assemble_coo!(
  I::AbstractArray{<:AbstractParamContainer},
  J::AbstractArray{<:AbstractParamContainer},
  V::AbstractArray{<:AbstractParamContainer},
  row_partition)
  """
    Returns three JaggedArrays with the coo triplets
    to be sent to the corresponding owner parts in parts_snd
  """
  function setup_snd(part,parts_snd,row_lids,coo_values)
    global_to_local_row = global_to_local(row_lids)
    local_row_to_owner = local_to_owner(row_lids)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_gi, k_gj, k_v = coo_values
    k_gi1, k_gj1 = first(k_gi), first(k_gj)
    for k in 1:length(k_gi)
      gi = k_gi1[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    gi_snd_data = zeros(eltype(k_gi1),ptrs[end]-1)
    gj_snd_data = zeros(eltype(k_gj1),ptrs[end]-1)
    v_snd_data = zeros(eltype(k_v),ptrs[end]-1)
    pv_snd_data = allocate_param_array(v_snd_data,length(k_v))
    for k in 1:length(k_gi1)
      gi = k_gi1[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        gj = k_gj1[k]
        p = ptrs[owner_to_i[owner]]
        gi_snd_data[p] = gi
        gj_snd_data[p] = gj
        for i = eachindex(pv_snd_data)
          v = k_v[i][k]
          pv_snd_data[i][p]  = v
          k_v[i][k] = zero(v)
        end
        ptrs[owner_to_i[owner]] += 1
      end
    end
    rewind_ptrs!(ptrs)
    pgi_snd_data = allocate_param_array(gi_snd_data,length(k_gi))
    pgj_snd_data = allocate_param_array(gj_snd_data,length(k_gj))
    gi_snd = JaggedArray(pgi_snd_data,ptrs)
    gj_snd = JaggedArray(pgj_snd_data,ptrs)
    v_snd = JaggedArray(pv_snd_data,ptrs)
    gi_snd, gj_snd, v_snd
  end
  """
    Pushes to coo_values the triplets gi_rcv,gj_rcv,v_rcv
    received from remote processes
  """
  function setup_rcv!(coo_values,gi_rcv,gj_rcv,v_rcv)
    k_gi, k_gj, k_v = coo_values
    current_n = length(first(k_gi))
    new_n = current_n + length(first(gi_rcv).data)
    resize!(k_gi,new_n)
    resize!(k_gj,new_n)
    resize!(k_v,new_n)
    for p in 1:length(first(gi_rcv).data)
      for i = eachindex(k_v)
        k_gi[i][current_n+p] = gi_rcv.data[i][p]
        k_gj[i][current_n+p] = gj_rcv.data[i][p]
        k_v[i][current_n+p] = v_rcv.data[i][p]
      end
    end
  end
  part = linear_indices(row_partition)
  parts_snd, parts_rcv = assembly_neighbors(row_partition)
  coo_values = map(tuple,I,J,V)
  gi_snd, gj_snd, v_snd = map(setup_snd,part,parts_snd,row_partition,coo_values) |> tuple_of_arrays
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

function GridapDistributed.assemble_coo_with_column_owner!(
  I::AbstractArray{<:AbstractParamContainer},
  J::AbstractArray{<:AbstractParamContainer},
  V::AbstractArray{<:AbstractParamContainer},
  row_partition,
  Jown::AbstractArray{<:AbstractParamContainer})
  """
    Returns three (Param)JaggedArrays with the coo triplets
    to be sent to the corresponding owner parts in parts_snd
  """
  function setup_snd(part,parts_snd,row_lids,coo_entries_with_column_owner)
    global_to_local_row = global_to_local(row_lids)
    local_row_to_owner = local_to_owner(row_lids)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_gi, k_gj, k_jo, k_v = coo_entries_with_column_owner
    k_gi1, k_jo1, k_gj1 = first(k_gi), first(k_jo), first(k_gj)
    for k in 1:length(k_gi1)
      gi = k_gi1[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    PartitionedArrays.length_to_ptrs!(ptrs)
    gi_snd_data = zeros(eltype(k_gi1),ptrs[end]-1)
    gj_snd_data = zeros(eltype(k_gj1),ptrs[end]-1)
    jo_snd_data = zeros(eltype(k_jo1),ptrs[end]-1)
    v_snd_data = zeros(eltype(k_v),ptrs[end]-1)
    pv_snd_data = allocate_param_array(v_snd_data,length(k_v))
    for k in 1:length(k_gi1)
      gi = k_gi1[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        gj = k_gj1[k]
        p = ptrs[owner_to_i[owner]]
        gi_snd_data[p] = gi
        gj_snd_data[p] = gj
        jo_snd_data[p] = k_jo1[k]
        for i = eachindex(pv_snd_data)
          v = k_v[i][k]
          pv_snd_data[i][p]  = v
          k_v[i][k] = zero(v)
        end
        ptrs[owner_to_i[owner]] += 1
      end
    end
    PartitionedArrays.rewind_ptrs!(ptrs)
    pgi_snd_data = allocate_param_array(gi_snd_data,length(k_gi))
    pgj_snd_data = allocate_param_array(gj_snd_data,length(k_gj))
    pjo_snd_data = allocate_param_array(jo_snd_data,length(k_jo))
    gi_snd = JaggedArray(pgi_snd_data,ptrs)
    gj_snd = JaggedArray(pgj_snd_data,ptrs)
    jo_snd = JaggedArray(pjo_snd_data,ptrs)
    v_snd = JaggedArray(pv_snd_data,ptrs)
    gi_snd, gj_snd, jo_snd, v_snd
  end
  """
    Pushes to coo_entries_with_column_owner the tuples
    gi_rcv,gj_rcv,jo_rcv,v_rcv received from remote processes
  """
  function setup_rcv!(coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
    k_gi, k_gj, k_jo, k_v = coo_entries_with_column_owner
    current_n = length(first(k_gi))
    new_n = current_n + length(first(gi_rcv).data)
    resize!(k_gi,new_n)
    resize!(k_gj,new_n)
    resize!(k_jo,new_n)
    resize!(k_v,new_n)
    for p in 1:length(first(gi_rcv).data)
      for i = eachindex(k_v)
        k_gi[i][current_n+p] = gi_rcv.data[i][p]
        k_gj[i][current_n+p] = gj_rcv.data[i][p]
        k_jo[i][current_n+p] = jo_rcv.data[i][p]
        k_v[i][current_n+p] = v_rcv.data[i][p]
      end
    end
  end
  part = linear_indices(row_partition)
  parts_snd, parts_rcv = assembly_neighbors(row_partition)
  coo_entries_with_column_owner = map(tuple,I,J,Jown,V)
  gi_snd, gj_snd, jo_snd, v_snd = map(setup_snd,part,parts_snd,row_partition,coo_entries_with_column_owner) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t1 = exchange(gi_snd,graph)
  t2 = exchange(gj_snd,graph)
  t3 = exchange(jo_snd,graph)
  t4 = exchange(v_snd,graph)
  @async begin
      gi_rcv = fetch(t1)
      gj_rcv = fetch(t2)
      jo_rcv = fetch(t3)
      v_rcv = fetch(t4)
      map(setup_rcv!,coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
      I,J,Jown,V
  end
end

function GridapDistributed._setup_prange(
  dofs_gids_prange::PRange,
  gids::AbstractVector{<:AbstractParamContainer};
  ghost=true,owners=nothing,kwargs...)

  gids1 = map(first,local_views(gids))
  if !ghost
    GridapDistributed._setup_prange_without_ghosts(dofs_gids_prange)
  elseif isa(owners,Nothing)
    GridapDistributed._setup_prange_with_ghosts(dofs_gids_prange,gids1)
  else
    if isa(owners,AbstractVector{<:AbstractParamContainer})
      owners1 = map(first,local_views(owners))
    else
      owners1 = owners
    end
    GridapDistributed._setup_prange_with_ghosts(dofs_gids_prange,gids1,owners1)
  end
end
