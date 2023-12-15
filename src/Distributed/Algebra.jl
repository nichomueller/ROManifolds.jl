function FEM.get_assembler(
  a::DistributedSparseMatrixAssembler,dc::DomainContribution,μ,t)
  @unpack (
    strategy,
    assems,
    matrix_builder,
    vector_builder,
    test_dofs_gids_prange,
    trial_dofs_gids_prange) = a
  _assems = map(assems) do assem
    get_assembler(assem,dc,μ,t)
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
  builder::PPTSparseMatrixBuilderCOO{A},axs::Tuple{<:PRange,<:PRange}) where A
  test_dofs_gids_prange,trial_dofs_gids_prange = axs
  counters = map(partition(test_dofs_gids_prange),partition(trial_dofs_gids_prange)) do r,c
    axs = (Base.OneTo(local_length(r)),Base.OneTo(local_length(c)))
    Algebra.CounterCOO{A}(axs)
  end
  #PTDistributedCounterCOO(builder.par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
  DistributedCounterCOO(builder.par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
end

function Algebra.get_array_type(::PPTSparseMatrixBuilderCOO{Tv}) where Tv
  @notimplemented
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
  #PPTVectorCounter(builder.par_strategy,counters,rows)
  PVectorCounter(builder.par_strategy,counters,rows)
end

function Algebra.get_array_type(::PPTVectorBuilder{Tv}) where Tv
  @notimplemented
end

# struct PTDistributedCounterCOO{A,B,C,D} <: GridapType
#   par_strategy::A
#   counters::B
#   test_dofs_gids_prange::C
#   trial_dofs_gids_prange::D
#   function PTDistributedCounterCOO(
#     par_strategy,
#     counters::AbstractArray{<:Algebra.CounterCOO},
#     test_dofs_gids_prange::PRange,
#     trial_dofs_gids_prange::PRange)
#     A = typeof(par_strategy)
#     B = typeof(counters)
#     C = typeof(test_dofs_gids_prange)
#     D = typeof(trial_dofs_gids_prange)
#     new{A,B,C,D}(par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
#   end
# end

# function GridapDistributed.local_views(a::PTDistributedCounterCOO)
#   a.counters
# end

# function GridapDistributed.local_views(
#   a::PTDistributedCounterCOO,
#   test_dofs_gids_prange,
#   trial_dofs_gids_prange)

#   @check test_dofs_gids_prange === a.test_dofs_gids_prange
#   @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
#   a.counters
# end

# function Algebra.nz_allocation(a::PTDistributedCounterCOO)
#   allocs = map(nz_allocation,a.counters)
#   PTDistributedCounterCOO(
#     a.par_strategy,
#     allocs,
#     a.test_dofs_gids_prange,
#     a.trial_dofs_gids_prange)
# end

# struct PPTVectorCounter{A,B,C}
#   par_strategy::A
#   counters::B
#   test_dofs_gids_prange::C
# end

# Algebra.LoopStyle(::Type{<:PPTVectorCounter}) = DoNotLoop()

# function GridapDistributed.local_views(a::PPTVectorCounter)
#   a.counters
# end

# function GridapDistributed.local_views(a::PPTVectorCounter,rows)
#   @check rows === a.test_dofs_gids_prange
#   a.counters
# end

# function Arrays.nz_allocation(a::PPTVectorCounter{<:FullyAssembledRows})
#   dofs = a.test_dofs_gids_prange
#   values = map(nz_allocation,a.counters)
#   PVectorAllocationTrackOnlyValues(a.par_strategy,values,dofs)
# end

# function Arrays.nz_allocation(
#   a::PTDistributedCounterCOO{<:SubAssembledRows},
#   b::PPTVectorCounter{<:SubAssembledRows})
#   A = nz_allocation(a)
#   dofs = b.test_dofs_gids_prange
#   values = map(nz_allocation,b.counters)
#   B = PVectorAllocationTrackOnlyValues(b.par_strategy,values,dofs)
#   return A,B
# end

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

# function Base.getproperty(
#   a::PVectorAllocationTrackOnlyValues{A,<:AbstractVector{<:PTInserter}} where A,
#   x::Symbol)

#   if x == :values
#     values = getfield(a,x)
#     map(x->getfield(x,:inserters),values)
#   elseif x in (:par_strategy,:test_dofs_gids_prange)
#     getfield(a,x)
#   else
#     @unreachable
#   end
# end

# function Base.getproperty(
#   a::PVectorAllocationTrackTouchedAndValues{A,<:AbstractVector{<:PTInserter}} where A,
#   x::Symbol)

#   if x == :values
#     values = getfield(a,x)
#     map(x->getfield(x,:inserters),values)
#   elseif x in (:allocations,:test_dofs_gids_prange)
#     getfield(a,x)
#   else
#     @unreachable
#   end
# end

# function Base.getproperty(
#   a::ArrayAllocationTrackTouchedAndValues{<:AbstractVector{<:PTInserter}},
#   x::Symbol)

#   @check x == :values
#   values = getfield(a,x)
#   map(x->getfield(x,:inserters),values)
# end

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
