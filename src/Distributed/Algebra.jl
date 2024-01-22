function FEM.get_passembler(
  a::DistributedSparseMatrixAssembler,dc::DomainContribution,μ,t)
  @unpack (
    strategy,
    assems,
    matrix_builder,
    vector_builder,
    test_dofs_gids_prange,
    trial_dofs_gids_prange) = a
  _assems = map(assems) do assem
    get_passembler(assem,dc,μ,t)
  end
  return DistributedSparseMatrixAssembler(
    strategy,
    _assems,
    matrix_builder,
    vector_builder,
    test_dofs_gids_prange,
    trial_dofs_gids_prange)
end

function FEM.PTSparseMatrixAssembler(a::DistributedSparseMatrixAssembler,μ,t)
  len = FEM._length(μ,t)
  local_mat_builder = SparseMatrixBuilder(a.matrix_builder.local_matrix_type)
  pt_matrix_builder = FEM.SparsePTMatrixBuilder(local_mat_builder,len)
  d_pt_matrix_builder = PPTSparseMatrixBuilderCOO(pt_matrix_builder,a.strategy)
  local_vec_builder = ArrayBuilder(a.vector_builder.local_vector_type)
  pt_vector_builder = FEM.PTArrayBuilder(local_vec_builder,len)
  d_pt_vector_builder = PPTVectorBuilder(pt_vector_builder,a.strategy)
  DistributedSparseMatrixAssembler(
    a.strategy,
    a.assems,
    d_pt_matrix_builder,
    d_pt_vector_builder,
    a.test_dofs_gids_prange,
    a.trial_dofs_gids_prange)
end

struct PPTSparseMatrixBuilderCOO{T,B}
  local_matrix::T
  par_strategy::B
end

function Algebra.nz_counter(
  builder::PPTSparseMatrixBuilderCOO{<:SparsePTMatrixBuilder{<:SparseMatrixBuilder{A}}},
  axs::Tuple{<:PRange,<:PRange}) where A
  b = builder.local_matrix
  test_dofs_gids_prange,trial_dofs_gids_prange = axs
  counters = map(partition(test_dofs_gids_prange),partition(trial_dofs_gids_prange)) do r,c
    axs = (Base.OneTo(local_length(r)),Base.OneTo(local_length(c)))
    counter = Algebra.CounterCOO{A}(axs)
    PTCounter(counter,b.length)
  end
  DistributedPTCounterCOO(builder.par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
end

function Algebra.get_array_type(::PPTSparseMatrixBuilderCOO{Tv}) where Tv
  @notimplemented
end

struct DistributedPTCounterCOO{A,B,C,D} <: GridapType
  par_strategy::A
  counters::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedPTCounterCOO(
    par_strategy,
    counters::AbstractArray{<:PTCounter},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(counters)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function GridapDistributed.local_views(a::DistributedPTCounterCOO)
  a.counters
end

function GridapDistributed.local_views(
  a::DistributedPTCounterCOO,test_dofs_gids_prange,trial_dofs_gids_prange)
  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  a.counters
end

function Algebra.nz_allocation(a::DistributedPTCounterCOO)
  allocs = map(nz_allocation,a.counters)
  DistributedPTAllocationCOO(a.par_strategy,allocs,a.test_dofs_gids_prange,a.trial_dofs_gids_prange)
end

struct DistributedPTAllocationCOO{A,B,C,D} <: GridapType
  par_strategy::A
  allocs::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedPTAllocationCOO(
    par_strategy,
    allocs::AbstractArray{<:PTAllocationCOO},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(allocs)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,allocs,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function GridapDistributed.change_axes(a::PTAllocationCOO,axes)
  PTAllocationCOO(change_axes(a.allocation,axes),a.length)
end

function GridapDistributed.change_axes(
  a::DistributedPTAllocationCOO{A,B,<:PRange,<:PRange},
  axes::Tuple{<:PRange,<:PRange}) where {A,B}
  local_axes = map(partition(axes[1]),partition(axes[2])) do rows,cols
    (Base.OneTo(local_length(rows)), Base.OneTo(local_length(cols)))
  end
  allocs = map(change_axes,a.allocs,local_axes)
  DistributedPTAllocationCOO(a.par_strategy,allocs,axes[1],axes[2])
end

function GridapDistributed.change_axes(
  a::MatrixBlock{<:DistributedPTAllocationCOO},
  axes::Tuple{<:Vector,<:Vector})
  block_ids  = CartesianIndices(a.array)
  rows, cols = axes
  array = map(block_ids) do I
    change_axes(a[I],(rows[I[1]],cols[I[2]]))
  end
  return ArrayBlock(array,a.touched)
end

function GridapDistributed.local_views(a::DistributedPTAllocationCOO)
  a.allocs
end

function GridapDistributed.local_views(
  a::DistributedPTAllocationCOO,test_dofs_gids_prange,trial_dofs_gids_prange)
  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  map(a.allocs) do alloc
    alloc.allocation
  end
end

function GridapDistributed.local_views(a::MatrixBlock{<:DistributedPTAllocationCOO})
  array = map(local_views,a.array) |> to_parray_of_arrays
  return map(ai -> ArrayBlock(ai,a.touched),array)
end

function GridapDistributed.get_allocations(a::DistributedPTAllocationCOO)
  I,J,V = map(local_views(a)) do alloc
    _alloc = alloc.allocation
    _alloc.I,_alloc.J,_alloc.V
  end |> tuple_of_arrays
  return I,J,V
end

function GridapDistributed.get_allocations(a::ArrayBlock{<:DistributedPTAllocationCOO})
  tuple_of_array_of_parrays = map(get_allocations,a.array) |> tuple_of_arrays
  return tuple_of_array_of_parrays
end

GridapDistributed.get_test_gids(a::DistributedPTAllocationCOO)  = a.test_dofs_gids_prange
GridapDistributed.get_trial_gids(a::DistributedPTAllocationCOO) = a.trial_dofs_gids_prange
GridapDistributed.get_test_gids(a::ArrayBlock{<:DistributedPTAllocationCOO})  = map(get_test_gids,diag(a.array))
GridapDistributed.get_trial_gids(a::ArrayBlock{<:DistributedPTAllocationCOO}) = map(get_trial_gids,diag(a.array))

function Algebra.create_from_nz(a::DistributedPTAllocationCOO{<:FullyAssembledRows})
  f(x) = nothing
  A, = GridapDistributed._fa_create_from_nz_with_callback(f,a)
  return A
end

function Algebra.create_from_nz(a::ArrayBlock{<:DistributedPTAllocationCOO{<:FullyAssembledRows}})
  f(x) = nothing
  A, = GridapDistributed._fa_create_from_nz_with_callback(f,a)
  return A
end

function Algebra.create_from_nz(a::DistributedPTAllocationCOO{<:SubAssembledRows})
  f(x) = nothing
  A, = GridapDistributed._sa_create_from_nz_with_callback(f,f,a)
  return A
end

function Algebra.create_from_nz(a::ArrayBlock{<:DistributedPTAllocationCOO{<:SubAssembledRows}})
  f(x) = nothing
  A, = GridapDistributed._sa_create_from_nz_with_callback(f,f,a)
  return A
end

function Algebra.create_from_nz(
  a::DistributedPTAllocationCOO{<:FullyAssembledRows},
  c_fespace::PVectorAllocationTrackOnlyValues{<:FullyAssembledRows})

  function callback(rows)
    _rhs_callback(c_fespace,rows)
  end

  A,b = GridapDistributed._fa_create_from_nz_with_callback(callback,a)
  return A,b
end

function Algebra.create_from_nz(
  a::DistributedPTAllocationCOO{<:SubAssembledRows},
  c_fespace::PVectorAllocationTrackOnlyValues{<:SubAssembledRows})

  function callback(rows)
    _rhs_callback(c_fespace,rows)
  end

  function async_callback(b)
    assemble!(b)
  end

  A,b = GridapDistributed._sa_create_from_nz_with_callback(callback,async_callback,a)
  return A,b
end

struct PPTVectorBuilder{T,B}
  local_vector::T
  par_strategy::B
end

function Algebra.nz_counter(builder::PPTVectorBuilder,axs::Tuple{<:PRange})
  b = builder.local_vector
  rows, = axs
  counters = map(partition(rows)) do rows
    axs = (Base.OneTo(local_length(rows)),)
    nz_counter(b,axs)
  end
  PVectorCounter(builder.par_strategy,counters,rows)
end

function Algebra.get_array_type(::PPTVectorBuilder{Tv}) where Tv
  @notimplemented
end

function Arrays.nz_allocation(
  a::PVectorCounter{<:SubAssembledRows,<:AbstractVector{<:PTCounter}})
  dofs = a.test_dofs_gids_prange
  values = map(nz_allocation,a.counters)
  touched = map(values) do values
    fill!(Vector{Bool}(undef,length(first(values.inserters))),false)
  end
  allocations = map(values,touched) do values,touched
    ArrayAllocationTrackTouchedAndValues(touched,values)
  end
  return PVectorAllocationTrackTouchedAndValues(allocations,values,dofs)
end

for T in (
  :(PVectorAllocationTrackOnlyValues{A,<:AbstractVector{<:PTInserter}} where A),
  :(PVectorAllocationTrackTouchedAndValues{A,<:AbstractVector{<:PTInserter}} where A),
  :(ArrayAllocationTrackTouchedAndValues{<:AbstractVector{<:PTInserter}})
  )
  @eval begin
    function GridapDistributed._rhs_callback(a::$T,rows)
      values = a.values
      ptvalues = map(values) do v
        v.inserters
      end
      b_fespace = PVector(ptvalues,partition(a.test_dofs_gids_prange))
      b = similar(b_fespace,eltype(b_fespace),(rows,))
      b .= b_fespace

      function transfer_ghost(b,b_fespace,ids,ids_fespace)
        num_ghosts_vec = ghost_length(ids)
        gho_to_loc_vec = ghost_to_local(ids)
        loc_to_glo_vec = local_to_global(ids)
        gid_to_lid_fe  = global_to_local(ids_fespace)
        for ghost_lid_vec in 1:num_ghosts_vec
          lid_vec = gho_to_loc_vec[ghost_lid_vec]
          gid = loc_to_glo_vec[lid_vec]
          lid_fespace = gid_to_lid_fe[gid]
          for k in eachindex(b)
            b[k][lid_vec] = b_fespace[k][lid_fespace]
          end
        end
      end
      map(
        transfer_ghost,
        partition(b),
        partition(b_fespace),
        b.index_partition,
        b_fespace.index_partition)

      return b
    end
  end
end
