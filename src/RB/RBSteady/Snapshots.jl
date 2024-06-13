abstract type AbstractSnapshots{T,N,L,D,I<:AbstractIndexMap{D},R<:AbstractParamRealization} <: AbstractParamContainer{T,N,L} end

ParamDataStructures.get_values(s::AbstractSnapshots) = @abstractmethod
IndexMaps.get_index_map(s::AbstractSnapshots) = @abstractmethod
get_realization(s::AbstractSnapshots) = @abstractmethod

num_space_dofs(s::AbstractSnapshots) = size(get_index_map(s))
ParamDataStructures.num_params(s::AbstractSnapshots) = num_params(get_realization(s))

function Snapshots(s::AbstractArray,i::AbstractIndexMap,r::AbstractParamRealization)
  @abstractmethod
end

function IndexMaps.change_index_map(f,s::AbstractSnapshots)
  index_map′ = change_index_map(f,get_index_map(s))
  Snapshots(get_values(s),get_realization(s),index_map′)
end

function flatten_snapshots(s::AbstractSnapshots)
  change_index_map(TrivialIndexMap,s)
end

abstract type AbstractSteadySnapshots{T,N,L,D,I,R<:ParamRealization} <: AbstractSnapshots{T,N,L,D,I,R} end

Base.size(s::AbstractSteadySnapshots) = (num_space_dofs(s)...,num_params(s))

struct BasicSnapshots{T,N,L,D,I,R,A} <: AbstractSteadySnapshots{T,N,L,D,I,R}
  data::A
  index_map::I
  realization::R
  function BasicSnapshots(
    data::A,
    index_map::I,
    realization::R
    ) where {T,N,L,A<:AbstractParamArray{T,N,L}}
    new{T,N+1,L,I,R,A}(data,index_map,realization)
  end
end

function Snapshots(s::AbstractParamArray,i::AbstractIndexMap,r::ParamRealization)
  BasicSnapshots(s,i,r)
end

ParamDataStructures.get_values(s::BasicSnapshots) = s.data
IndexMaps.get_index_map(s::BasicSnapshots) = s.index_map
get_realization(s::BasicSnapshots) = s.realization

Base.@propagate_inbounds function Base.getindex(
  s::BasicSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data,iparam)
  getindex(sparam,ispace′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::BasicSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_view(s.data,iparam)
  setindex!(sparam,v,ispace′)
end

struct SnapshotsAtIndices{T,N,L,D,I,R,A<:AbstractSteadySnapshots{T,N,L,D,I,R},B<:NTuple{N,AbstractUnitRange{Int}}} <: AbstractSteadySnapshots{T,N,L,D,I,R}
  snaps::A
  indices::B
end

function SnapshotsAtIndices(s::SnapshotsAtIndices,indices)
  new_srange,new_prange = indices
  old_srange,old_prange = s.indices
  @check intersect(old_srange,new_srange) == new_srange
  @check intersect(old_prange,new_prange) == new_prange
  SnapshotsAtIndices(s.snaps,indices)
end

space_indices(s::SnapshotsAtIndices{T,N}) where {T,N} = s.selected_indices[1:N-1]
param_indices(s::SnapshotsAtIndices) = s.selected_indices[N]
num_space_dofs(s::SnapshotsAtIndices) = length.(space_indices(s))

ParamDataStructures.get_values(s::SnapshotsAtIndices) = get_values(s.snaps)
IndexMaps.get_index_map(s::SnapshotsAtIndices) = get_index_map(s.snaps)
get_realization(s::SnapshotsAtIndices) = get_realization(s.snaps)

Base.@propagate_inbounds function Base.getindex(
  s::SnapshotsAtIndices{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = getindex.(space_indices(s),ispace)
  iparam′ = getindex(param_indices(s),iparam)
  getindex(s.snaps,ispace′...,iparam′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::SnapshotsAtIndices{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace′ = getindex.(space_indices(s),ispace)
  iparam′ = getindex(param_indices(s),iparam)
  setindex!(s.snaps,v,ispace′...,iparam′)
end

format_range(a::Number,l::Int) = Base.OneTo(a)
format_range(a::Colon,l::Int) = Base.OneTo(l)

function select_snapshots(
  s::AbstractSteadySnapshots,
  spacerange::Tuple{Vararg{AbstractUnitRange}},
  paramrange::AbstractUnitRange)

  indices = (spacerange,paramrange)
  SnapshotsAtIndices(s,indices)
end

function select_snapshots(s::AbstractSteadySnapshots,spacerange,paramrange)
  srange = Tuple.(format_range(spacerange,num_space_dofs(s)))
  prange = format_range(paramrange,num_params(s))
  select_snapshots(s,srange,prange)
end

function select_snapshots(
  s::AbstractSteadySnapshots,
  paramrange;
  spacerange=Base.OneTo.(num_space_dofs(s)))

  select_snapshots(s,spacerange,paramrange)
end

function select_snapshots_entries(s::AbstractSteadySnapshots,spacerange)
  ss = select_snapshots(s,spacerange,Base.OneTo(num_params(s)))
  data = collect(ss)
  return ArrayOfArrays(data)
end

const SparseSnapshots{T,N,L,D,I,R,A<:MatrixOfSparseMatricesCSC} = BasicSnapshots{T,N,L,D,I,R,A}

function ParamDataStructures.recast(s::SparseSnapshots,a::AbstractMatrix)
  A = param_getindex(s.data,1)
  return recast(A,a)
end

const StandardSnapshots{T,L,I,R} = AbstractSnapshots{T,2,L,1,I,R}
const StandardSteadySnapshots{T,L,I,R} = AbstractSteadySnapshots{T,2,L,1,I,R}
const StandardSparseSnapshots{T,L,I,R,A} = SparseSnapshots{T,2,L,1,I,R,A}

struct BlockSnapshots{S,N,L} <: AbstractParamContainer{S,N,L}
  array::Array{S,N}
  touched::Array{Bool,N}

  function BlockSnapshots(
    array::Array{S,N},
    touched::Array{Bool,N}
    ) where {S<:AbstractSnapshots,N}

    @check size(array) == size(touched)
    new{S,N}(array,touched)
  end
end

function BlockSnapshots(k::BlockMap{N},a::AbstractArray{S}) where {S<:AbstractSnapshots,N}
  array = Array{S,N}(undef,k.size)
  touched = fill(false,k.size)
  for (t,i) in enumerate(k.indices)
    array[i] = a[t]
    touched[i] = true
  end
  BlockSnapshots(array,touched)
end

function Fields.BlockMap(s::NTuple,inds::Vector{<:Integer})
  cis = [CartesianIndex((i,)) for i in inds]
  BlockMap(s,cis)
end

function Snapshots(data::BlockArrayOfArrays,i::AbstractIndexMap,r::AbstractParamRealization)
  block_values = blocks(data)
  nblocks = blocksize(data)
  active_block_ids = findall(!iszero,block_values)
  block_map = BlockMap(nblocks,active_block_ids)
  active_block_snaps = [Snapshots(block_values[n],i,r) for n in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps...)
end

Base.size(s::BlockSnapshots) = size(s.array)

function Base.getindex(s::BlockSnapshots,i...)
  if !s.touched[i...]
    return nothing
  end
  s.array[i...]
end

function Base.setindex!(s::BlockSnapshots,v,i...)
  @check s.touched[i...] "Only touched entries can be set"
  s.array[i...] = v
end

function Arrays.testitem(s::BlockSnapshots)
  i = findall(s.touched)
  if length(i) != 0
    s.array[i[1]]
  else
    error("This block snapshots structure is empty")
  end
end

get_touched_blocks(s) = @abstractmethod
get_touched_blocks(s::ArrayBlock) = findall(s.touched)
get_touched_blocks(s::BlockSnapshots) = findall(s.touched)

IndexMaps.get_index_map(s::BlockSnapshots) = get_index_map(testitem(s))
get_realization(s::BlockSnapshots) = get_realization(testitem(s))
