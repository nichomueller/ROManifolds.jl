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
