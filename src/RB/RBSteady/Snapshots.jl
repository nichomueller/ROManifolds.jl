"""
    abstract type AbstractSnapshots{T,N,D,I<:AbstractDofMap{D},R<:AbstractRealization,A}
      <: AbstractParamContainer{T,N} end

Type representing a collection of parametric abstract arrays of eltype T,
that are associated with a realization of type R. The (spatial)
entries of any instance of AbstractSnapshots are indexed according to an index
map of type I<:AbstractDofMap{D}, where D encodes the spatial dimension.

Subtypes:
- [`AbstractSteadySnapshots`](@ref)
- [`AbstractTransientSnapshots`](@ref)

"""
abstract type AbstractSnapshots{T,N,D,I<:AbstractDofMap{D},R<:AbstractRealization,A} <: AbstractParamContainer{T,N} end

Utils.get_values(s::AbstractSnapshots) = @abstractmethod
get_indexed_data(s::AbstractSnapshots) = @abstractmethod
DofMaps.get_dof_map(s::AbstractSnapshots) = @abstractmethod
get_realization(s::AbstractSnapshots) = @abstractmethod

ParamDataStructures.param_length(s::AbstractSnapshots) = @notimplemented

"""
    num_space_dofs(s::AbstractSnapshots{T,N,D}) where {T,N,D} -> NTuple{D,Integer}

Returns the spatial size of the snapshots

"""
num_space_dofs(s::AbstractSnapshots) = size(get_dof_map(s))
ParamDataStructures.num_params(s::AbstractSnapshots) = num_params(get_realization(s))

"""
    Snapshots(s::AbstractArray,i::AbstractDofMap,r::AbstractRealization
      ) -> AbstractSnapshots

Constructor of an instance of AbstractSnapshots

"""
function Snapshots(s::AbstractArray,i::AbstractDofMap,r::AbstractRealization)
  @abstractmethod
end

function DofMaps.recast(a::AbstractArray,s::AbstractSnapshots)
  return recast(a,get_dof_map(s))
end

"""
    flatten_snapshots(s::AbstractSnapshots) -> AbstractSnapshots

The output snapshots are indexed according to a [`TrivialDofMap`](@ref)

"""
function flatten_snapshots(s::AbstractSnapshots)
  i′ = TrivialDofMap(get_dof_map(s))
  Snapshots(get_values(s),i′,get_realization(s))
end

"""
    abstract type AbstractSteadySnapshots{T,N,D,I,A}
      <: AbstractSnapshots{T,N,D,I,<:Realization,A} end

Spatial specialization of an [`AbstractSnapshots`](@ref). The dimension `N` of a
AbstractSteadySnapshots is equal to `D` + 1, where `D` represents the number of
spatial axes, to which a parametric dimension is added.

Subtypes:
- [`GenericSnapshots`](@ref).
- [`SnapshotsAtIndices`](@ref).

# Examples

```jldoctest
julia> ns1,ns2,np = 2,2,2
(2, 2, 2)
julia> data = [rand(ns1*ns2) for ip = 1:np]
2-element Vector{Vector{Float64}}:
 [0.4684452123483283, 0.1195886171030737, 0.1151790990455997, 0.0375575515915656]
 [0.9095165124078269, 0.7346081836882059, 0.8939511550403715, 0.2288086807377305]
julia> i = DofMap(collect(LinearIndices((ns1,ns2))))
2×2 DofMap{2, Int64}:
 1  3
 2  4
julia> pspace = ParamSpace(fill([0,1],3))
Set of parameters in [[0, 1], [0, 1], [0, 1]], sampled with UniformSampling()
julia> r = realization(pspace,nparams=np)
Realization{Vector{Vector{Float64}}}([
  [0.4021870679335007, 0.6585653527784044, 0.5110768420820191],
  [0.0950901750101361, 0.7049711670440882, 0.3490097863258958]])
julia> s = Snapshots(ParamArray(data),i,r)
2×2×2 GenericSnapshots{Float64, 3, 2, 2, DofMap{2, Int64},
  Realization{Vector{Vector{Float64}}}, VectorOfVectors{Float64, 2}}:
  [:, :, 1] =
  0.468445  0.115179
  0.119589  0.0375576

  [:, :, 2] =
  0.909517  0.893951
  0.734608  0.228809
```

"""
abstract type AbstractSteadySnapshots{T,N,D,I,R<:Realization,A} <: AbstractSnapshots{T,N,D,I,R,A} end

Base.size(s::AbstractSteadySnapshots) = (num_space_dofs(s)...,num_params(s))

"""
    struct GenericSnapshots{T,N,D,I,R,A} <: AbstractSteadySnapshots{T,N,D,I,R,A} end

Most standard implementation of a AbstractSteadySnapshots

"""
struct GenericSnapshots{T,N,D,I,R,A} <: AbstractSteadySnapshots{T,N,D,I,R,A}
  data::A
  dof_map::I
  realization::R

  function GenericSnapshots(
    data::A,
    dof_map::I,
    realization::R
    ) where {T,N,D,R,A<:AbstractParamArray{T,N},I<:AbstractDofMap{D}}

    new{T,D+1,D,I,R,A}(data,dof_map,realization)
  end
end

function Snapshots(s::AbstractParamArray,i::AbstractDofMap,r::Realization)
  GenericSnapshots(s,i,r)
end

ParamDataStructures.get_all_data(s::GenericSnapshots) = get_all_data(s.data)
Utils.get_values(s::GenericSnapshots) = s.data
DofMaps.get_dof_map(s::GenericSnapshots) = s.dof_map
get_realization(s::GenericSnapshots) = s.realization

function get_indexed_data(s::GenericSnapshots{T}) where T
  vi = vectorize(get_dof_map(s))
  data = get_all_data(s)
  if isnothing(findfirst(iszero,vi))
    return view(data,vi,:)
  end
  i = get_dof_map(s)
  idata = zeros(T,size(data))
  for (j,ij) in enumerate(i)
    for k in 1:num_params(s)
      if ij > 0
        @inbounds idata[ij,k] = data[j,k]
      end
    end
  end
  return idata
end

Base.@propagate_inbounds function Base.getindex(
  s::GenericSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = s.dof_map[ispace...]
  data = get_all_data(s)
  ispace′ == 0 ? zero(eltype(s)) : data[ispace′,iparam]
end

Base.@propagate_inbounds function Base.setindex!(
  s::GenericSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = s.dof_map[ispace...]
  data = get_all_data(s)
  ispace′ != 0 && (data[ispace′,iparam] = v)
end

"""
    struct SnapshotsAtIndices{T,N,D,I,R,A<:AbstractSteadySnapshots{T,N,D,I,R},B<:AbstractUnitRange{Int}}
      <: AbstractSteadySnapshots{T,N,D,I,R,A} end

Represents a AbstractSteadySnapshots `snaps` whose parametric range is restricted
to the indices in `prange`. This type essentially acts as a view for suptypes of
AbstractSteadySnapshots, at every space location, on a selected number of
parameter indices. An instance of SnapshotsAtIndices is created by calling the
function [`select_snapshots`](@ref)

"""
struct SnapshotsAtIndices{T,N,D,I,R,A<:AbstractSteadySnapshots{T,N,D,I,R},B<:AbstractUnitRange{Int}} <: AbstractSteadySnapshots{T,N,D,I,R,A}
  snaps::A
  prange::B
  function SnapshotsAtIndices(snaps::A,prange::B) where {T,N,D,I,R,A<:AbstractSteadySnapshots{T,N,D,I,R},B}
    @assert 1 <= minimum(prange) <= maximum(prange) <= _num_all_params(snaps)
    new{T,N,D,I,R,A,B}(snaps,prange)
  end
end

function SnapshotsAtIndices(s::SnapshotsAtIndices,prange)
  old_prange = s.prange
  @check intersect(old_prange,prange) == prange
  SnapshotsAtIndices(s.snaps,prange)
end

param_indices(s::SnapshotsAtIndices) = s.prange
ParamDataStructures.num_params(s::SnapshotsAtIndices) = length(param_indices(s))
ParamDataStructures.get_all_data(s::SnapshotsAtIndices) = get_all_data(s.snaps)
DofMaps.get_dof_map(s::SnapshotsAtIndices) = get_dof_map(s.snaps)

_num_all_params(s::AbstractSnapshots) = num_params(s)
_num_all_params(s::SnapshotsAtIndices) = _num_all_params(s.snaps)

function Utils.get_values(s::SnapshotsAtIndices)
  data = get_all_data(s)
  v = view(data,:,param_indices(s))
  ConsecutiveParamArray(v)
end

function get_indexed_data(s::SnapshotsAtIndices)
  idata = get_indexed_data(s.snaps)
  view(idata,:,param_indices(s))
end

get_realization(s::SnapshotsAtIndices) = get_realization(s.snaps)[s.prange]

Base.@propagate_inbounds function Base.getindex(
  s::SnapshotsAtIndices{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  iparam′ = getindex(param_indices(s),iparam)
  getindex(s.snaps,ispace...,iparam′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::SnapshotsAtIndices{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam
  iparam′ = getindex(param_indices(s),iparam)
  setindex!(s.snaps,v,ispace...,iparam′)
end

format_range(a::AbstractUnitRange,l::Int) = a
format_range(a::Base.OneTo{Int},l::Int) = 1:a.stop
format_range(a::Number,l::Int) = a:a
format_range(a::Colon,l::Int) = 1:l

"""
    select_snapshots(s::AbstractSteadySnapshots,prange) -> SnapshotsAtIndices
    select_snapshots(s::AbstractTransientSnapshots,trange,prange) -> TransientSnapshotsAtIndices

Restricts the parametric range of `s` to the indices `prange` steady cases, to
the indices `trange` and `prange` in transient cases, while leaving the spatial
entries intact. The restriction operation is lazy.

"""
function select_snapshots(s::AbstractSteadySnapshots,prange)
  prange = format_range(prange,num_params(s))
  SnapshotsAtIndices(s,prange)
end

struct ReshapedSnapshots{T,N,N′,D,I,R,A<:AbstractSteadySnapshots{T,N′,D,I,R},B} <: AbstractSteadySnapshots{T,N′,D,I,R,A}
  snaps::A
  size::NTuple{N,Int}
  mi::B
end

Base.size(s::ReshapedSnapshots) = s.size

function Base.reshape(s::AbstractSnapshots,dims::Dims)
  n = length(s)
  prod(dims) == n || DimensionMismatch()

  strds = Base.front(Base.size_to_strides(map(length,axes(s))..., 1))
  strds1 = map(s->max(1,Int(s)),strds)
  mi = map(Base.SignedMultiplicativeInverse,strds1)
  ReshapedSnapshots(s,dims,reverse(mi))
end

Base.@propagate_inbounds function Base.getindex(
  s::ReshapedSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ax = axes(s.snaps)
  i′ = Base.offset_if_vec(Base._sub2ind(size(s),i...),ax)
  i′′ = Base.ind2sub_rs(ax,s.mi,i′)
  Base._unsafe_getindex_rs(s.snaps,i′′)
end

function Base.setindex!(
  s::ReshapedSnapshots{T,N},
  v,i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ax = axes(s.snaps)
  i′ = Base.offset_if_vec(Base._sub2ind(size(s),i...),ax)
  s.snaps[Base.ind2sub_rs(ax,s.mi,i′)] = v
  v
end

get_realization(s::ReshapedSnapshots) = get_realization(s.snaps)
DofMaps.get_dof_map(s::ReshapedSnapshots) = get_dof_map(s.snaps)

function Utils.get_values(s::ReshapedSnapshots)
  v = get_values(s.snaps)
  reshape(v.data,s.size)
end

function get_indexed_data(s::ReshapedSnapshots)
  v = get_indexed_data(s.snaps)
  reshape(v,s.size)
end

function Base.:*(A::AbstractSnapshots{T,2},B::AbstractSnapshots{S,2}) where {T,S}
  consec_mul(get_indexed_data(A),get_indexed_data(B))
end

function Base.:*(A::AbstractSnapshots{T,2},B::Adjoint{S,<:AbstractSnapshots}) where {T,S}
  consec_mul(get_indexed_data(A),adjoint(get_indexed_data(B.parent)))
end

function Base.:*(A::AbstractSnapshots{T,2},B::AbstractMatrix{S}) where {T,S}
  consec_mul(get_indexed_data(A),B)
end

function Base.:*(A::AbstractSnapshots{T,2},B::Adjoint{T,<:AbstractMatrix{S}}) where {T,S}
  consec_mul(get_indexed_data(A),B)
end

function Base.:*(A::Adjoint{T,<:AbstractSnapshots{T,2}},B::AbstractSnapshots{S,2}) where {T,S}
  consec_mul(adjoint(get_indexed_data(A.parent)),get_indexed_data(B))
end

function Base.:*(A::AbstractMatrix{T},B::AbstractSnapshots{S,2}) where {T,S}
  consec_mul(A,get_indexed_data(B))
end

function Base.:*(A::Adjoint{T,<:AbstractMatrix},B::AbstractSnapshots{S,2}) where {T,S}
  consec_mul(A,get_indexed_data(B))
end

consec_mul(A::AbstractArray,B::AbstractArray) = A*B

for T in (:ConsecutiveParamArray,:ConsecutiveParamSparseMatrix)
  @eval begin
    consec_mul(A::$T,B::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}}) where S = get_all_data(A)*B
    consec_mul(A::Adjoint{S,<:$T},B::Union{<:AbstractArray,Adjoint{U,<:AbstractArray}}) where {S,U} = adjoint(get_all_data(A.parent))*B
    consec_mul(A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::$T) where S = A*get_all_data(B)
    consec_mul(A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::Adjoint{U,<:$T}) where {S,U} = A*adjoint(get_all_data(B.parent))
  end
end

# sparse interface

const SimpleSparseSnapshots{T,N,D,I,R,A<:ParamSparseMatrix} = AbstractSnapshots{T,N,D,I,R,A}
const CompositeSparseSnapshots{T,N,D,I,R,A<:SimpleSparseSnapshots} = AbstractSnapshots{T,N,D,I,R,A}
const GenericSparseSnapshots{T,N,D,I,R,A<:CompositeSparseSnapshots} = AbstractSnapshots{T,N,D,I,R,A}
const SparseSnapshots{T,N,D,I,R} = Union{
  SimpleSparseSnapshots{T,N,D,I,R},
  CompositeSparseSnapshots{T,N,D,I,R},
  GenericSparseSnapshots{T,N,D,I,R}
}

# multi field interface

"""
    struct BlockSnapshots{S,N} <: AbstractParamContainer{S,N}

Block container for AbstractSnapshots of type `S` in a MultiField setting. This
type is conceived similarly to [`ArrayBlock`](@ref) in [`Gridap`](@ref)

"""
struct BlockSnapshots{S,N} <: AbstractParamContainer{S,N}
  array::Array{S,N}
  touched::Array{Bool,N}
  function BlockSnapshots(array::Array{S,N},touched::Array{Bool,N}) where {S,N}
    @check size(array) == size(touched)
    new{S,N}(array,touched)
  end
end

function BlockSnapshots(k::BlockMap{N},a::AbstractArray{S}) where {S,N}
  array = Array{S,N}(undef,k.size)
  touched = fill(false,k.size)
  for (t,i) in enumerate(k.indices)
    array[i] = a[t]
    touched[i] = true
  end
  BlockSnapshots(array,touched)
end

function Fields.BlockMap(s::NTuple,inds::AbstractVector{<:Integer})
  cis = [CartesianIndex((i,)) for i in inds]
  BlockMap(s,cis)
end

function Snapshots(data::BlockParamArray,i::AbstractArray{<:AbstractDofMap},r::AbstractRealization)
  block_values = blocks(data)
  nblocks = blocksize(data)
  active_block_ids = findall(!iszero,block_values)
  block_map = BlockMap(nblocks,active_block_ids)
  active_block_snaps = [Snapshots(block_values[n],i[n],r) for n in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps)
end

BlockArrays.blocks(s::BlockSnapshots) = s.array

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

DofMaps.get_dof_map(s::BlockSnapshots) = get_dof_map(testitem(s))
get_realization(s::BlockSnapshots) = get_realization(testitem(s))

function Utils.get_values(s::BlockSnapshots)
  map(get_values,s.array) |> mortar
end

function get_indexed_data(s::BlockSnapshots)
  map(get_indexed_data,s.array)
end

for f in (:flatten_snapshots,:select_snapshots)
  @eval begin
    function Arrays.return_cache(::typeof($f),s::AbstractSnapshots,args...;kwargs...)
      $f(s,args...;kwargs...)
    end

    function Arrays.return_cache(::typeof($f),s::BlockSnapshots,args...;kwargs...)
      i = findfirst(s.touched)
      @notimplementedif isnothing(i)
      cache = return_cache($f,s[i],args...;kwargs...)
      block_cache = Array{typeof(cache),ndims(s)}(undef,size(s))
      return block_cache
    end
  end
end

for f in (:flatten_snapshots,:select_snapshots)
  @eval begin
    function $f(s::BlockSnapshots,args...;kwargs...)
      array = return_cache($f,s,args...;kwargs...)
      touched = s.touched
      for i in eachindex(touched)
        if touched[i]
          array[i] = $f(s[i],args...;kwargs...)
        end
      end
      return BlockSnapshots(array,touched)
    end
  end
end

# utils

function Snapshots(a::ArrayContribution,i::ArrayContribution,r::AbstractRealization)
  contribution(a.trians) do trian
    Snapshots(a[trian],i[trian],r)
  end
end

function select_snapshots(a::ArrayContribution,args...;kwargs...)
  contribution(a.trians) do trian
    select_snapshots(a[trian],args...;kwargs...)
  end
end
