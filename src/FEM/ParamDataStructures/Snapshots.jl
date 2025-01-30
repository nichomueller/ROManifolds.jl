"""
    abstract type AbstractSnapshots{T,N} <: AbstractParamContainer{T,N} end

Type representing N-dimensional arrays of snapshots. Subtypes must contain the
following information:

- data: a (parametric) array
- realization: a subtype of `AbstractRealization`, representing the points
  in the parameter space used to compute the array `data`
- dof map: a subtype of `AbstractDofMap`, representing a reindexing strategy
  for the array `data`

Subtypes:

- [`Snapshots`](@ref)
- [`BlockSnapshots`](@ref)
"""
abstract type AbstractSnapshots{T,N} <: AbstractParamContainer{T,N} end

Utils.get_values(s::AbstractSnapshots) = @abstractmethod

"""
    get_realization(s::AbstractSnapshots) -> AbstractRealization

Returns the realizations associated to the snapshots `s`
"""
get_realization(s::AbstractSnapshots) = @abstractmethod

DofMaps.get_dof_map(s::AbstractSnapshots) = @abstractmethod

"""
    abstract type Snapshots{T,N,D,I<:AbstractDofMap{D},R<:AbstractRealization,A}
      <: AbstractSnapshots{T,N} end

Type representing a collection of parametric abstract arrays of eltype `T`,
that are associated with a realization of type `R`. The (spatial)
entries of any instance of `Snapshots` are indexed according to an index
map of type `I`:AbstractDofMap{`D`}, where `D` encodes the spatial dimension. Note
that, as opposed to subtypes of `AbstractParamArray`, which are arrays
of arrays, subtypes of `Snapshots` are arrays of numbers.

Subtypes:

- [`SteadySnapshots`](@ref)
- [`TransientSnapshots`](@ref)
"""
abstract type Snapshots{T,N,D,I<:AbstractDofMap{D},R<:AbstractRealization,A} <: AbstractSnapshots{T,N} end

"""
    get_indexed_data(s::Snapshots) -> AbstractArray

Returns the data in `s` reindexed according to the indexing strategy provided in `s`.
Note: this function is not lazy
"""
get_indexed_data(s::Snapshots) = @abstractmethod

DofMaps.get_dof_map(s::Snapshots) = @abstractmethod

param_length(s::Snapshots) = @notimplemented

"""
    num_space_dofs(s::Snapshots{T,N,D}) where {T,N,D} -> NTuple{D,Integer}

Returns the spatial size of the snapshots
"""
num_space_dofs(s::Snapshots) = size(get_dof_map(s))

num_params(s::Snapshots) = num_params(get_realization(s))

function Snapshots(s::AbstractArray,i::AbstractDofMap,r::AbstractRealization)
  @abstractmethod
end

function DofMaps.recast(a::AbstractArray,s::Snapshots)
  return recast(a,get_dof_map(s))
end

function DofMaps.change_dof_map(s::Snapshots,args...)
  i′ = change_dof_map(get_dof_map(s),args...)
  return Snapshots(get_values(s),i′,get_realization(s))
end

function DofMaps.flatten(s::Snapshots)
  i′ = flatten(get_dof_map(s))
  Snapshots(get_values(s),i′,get_realization(s))
end

"""
    abstract type SteadySnapshots{T,N,D,I,A} <: Snapshots{T,N,D,I,<:Realization,A} end

Spatial specialization of an `Snapshots`. The dimension `N` of a
SteadySnapshots is equal to `D` + 1, where `D` represents the number of
spatial axes, to which a parametric dimension is added.

Subtypes:
- [`GenericSnapshots`](@ref)
- [`SnapshotsAtIndices`](@ref)
- [`ReshapedSnapshots`](@ref)
"""
abstract type SteadySnapshots{T,N,D,I,R<:Realization,A} <: Snapshots{T,N,D,I,R,A} end

Base.size(s::SteadySnapshots) = (num_space_dofs(s)...,num_params(s))

"""
    struct GenericSnapshots{T,N,D,I,R,A} <: SteadySnapshots{T,N,D,I,R,A}
      data::A
      dof_map::I
      realization::R
    end

Most standard implementation of a [`SteadySnapshots`](@ref)
"""
struct GenericSnapshots{T,N,D,I,R,A} <: SteadySnapshots{T,N,D,I,R,A}
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

get_all_data(s::GenericSnapshots) = get_all_data(s.data)
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
    struct SnapshotsAtIndices{T,N,D,I,R,A<:SteadySnapshots{T,N,D,I,R},B} <: SteadySnapshots{T,N,D,I,R,A}
      snaps::A
      prange::B
    end

Represents a SteadySnapshots `snaps` whose parametric range is restricted
to the indices in `prange`. This type essentially acts as a view for suptypes of
SteadySnapshots, at every space location, on a selected number of
parameter indices. An instance of SnapshotsAtIndices is created by calling the
function `select_snapshots`
"""
struct SnapshotsAtIndices{T,N,D,I,R,A<:SteadySnapshots{T,N,D,I,R},B<:AbstractUnitRange{Int}} <: SteadySnapshots{T,N,D,I,R,A}
  snaps::A
  prange::B
  function SnapshotsAtIndices(snaps::A,prange::B) where {T,N,D,I,R,A<:SteadySnapshots{T,N,D,I,R},B}
    @assert 1 <= minimum(prange) <= maximum(prange) <= _num_all_params(snaps)
    new{T,N,D,I,R,A,B}(snaps,prange)
  end
end

function SnapshotsAtIndices(s::SnapshotsAtIndices,prange)
  prange′ = s.prange[prange]
  SnapshotsAtIndices(s.snaps,prange′)
end

param_indices(s::SnapshotsAtIndices) = s.prange
num_params(s::SnapshotsAtIndices) = length(param_indices(s))
get_all_data(s::SnapshotsAtIndices) = get_all_data(s.snaps)
DofMaps.get_dof_map(s::SnapshotsAtIndices) = get_dof_map(s.snaps)

_num_all_params(s::Snapshots) = num_params(s)
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
    select_snapshots(s::SteadySnapshots,prange) -> SnapshotsAtIndices
    select_snapshots(s::TransientSnapshots,trange,prange) -> TransientSnapshotsAtIndices

Restricts the parametric range of `s` to the indices `prange` steady cases, to
the indices `trange` and `prange` in transient cases, while leaving the spatial
entries intact. The restriction operation is lazy.
"""
function select_snapshots(s::SteadySnapshots,prange)
  prange = format_range(prange,num_params(s))
  SnapshotsAtIndices(s,prange)
end

"""
    struct ReshapedSnapshots{T,N,N′,D,I,R,A<:SteadySnapshots{T,N′,D,I,R},B} <: SteadySnapshots{T,N,D,I,R,A}
      snaps::A
      size::NTuple{N,Int}
      mi::B
    end

Represents a SteadySnapshots `snaps` whose size is resized to `size`. This struct
is equivalent to `ReshapedArray`, and is only used to make sure the result
of this operation is still a subtype of SteadySnapshots
"""
struct ReshapedSnapshots{T,N,N′,D,I,R,A<:SteadySnapshots{T,N′,D,I,R},B} <: SteadySnapshots{T,N,D,I,R,A}
  snaps::A
  size::NTuple{N,Int}
  mi::B
end

Base.size(s::ReshapedSnapshots) = s.size

function Base.reshape(s::Snapshots,dims::Dims)
  n = length(s)
  prod(dims) == n || DimensionMismatch()

  strds = Base.front(Base.size_to_strides(map(length,axes(s))...,1))
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

# sparse interface

const SimpleSparseSnapshots{T,N,D,I,R,A<:ParamSparseMatrix} = Snapshots{T,N,D,I,R,A}
const CompositeSparseSnapshots{T,N,D,I,R,A<:SimpleSparseSnapshots} = Snapshots{T,N,D,I,R,A}
const GenericSparseSnapshots{T,N,D,I,R,A<:CompositeSparseSnapshots} = Snapshots{T,N,D,I,R,A}

"""
"""
const SparseSnapshots{T,N,D,I,R} = Union{
  SimpleSparseSnapshots{T,N,D,I,R},
  CompositeSparseSnapshots{T,N,D,I,R},
  GenericSparseSnapshots{T,N,D,I,R}
}

# multi field interface

"""
    struct BlockSnapshots{S<:Snapshots,N} <: AbstractSnapshots{S,N}
      array::Array{S,N}
      touched::Array{Bool,N}
    end

Block container for Snapshots of type `S` in a MultiField setting. This
type is conceived similarly to `ArrayBlock` in Gridap
"""
struct BlockSnapshots{S<:Snapshots,N} <: AbstractSnapshots{S,N}
  array::Array{S,N}
  touched::Array{Bool,N}

  function BlockSnapshots(
    array::Array{S,N},
    touched::Array{Bool,N}
    ) where {S<:Snapshots,N}

    @check size(array) == size(touched)
    new{S,N}(array,touched)
  end
end

function Snapshots(
  data::BlockParamArray{T,N},
  i::AbstractArray{<:AbstractDofMap},
  r::AbstractRealization) where {T,N}

  block_values = blocks(data)
  s = size(block_values)
  @check s == size(i)

  array = Array{Snapshots,N}(undef,s)
  touched = Array{Bool,N}(undef,s)
  for (j,dataj) in enumerate(block_values)
    if !iszero(dataj)
      array[j] = Snapshots(dataj,i[j],r)
      touched[j] = true
    else
      touched[j] = false
    end
  end

  BlockSnapshots(array,touched)
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

DofMaps.get_dof_map(s::BlockSnapshots) = map(get_dof_map,s.array)
get_realization(s::BlockSnapshots) = get_realization(testitem(s))

function Utils.get_values(s::BlockSnapshots)
  map(get_values,s.array) |> mortar
end

function get_indexed_data(s::BlockSnapshots)
  map(get_indexed_data,s.array)
end

for f in (:select_snapshots,:(DofMaps.flatten))
  @eval begin
    function Arrays.return_cache(::typeof($f),s::BlockSnapshots,args...;kwargs...)
      S = Snapshots
      N = ndims(s)
      block_cache = Array{S,N}(undef,size(s))
      return block_cache
    end

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

function Arrays.return_cache(::typeof(change_dof_map),s::BlockSnapshots,i::AbstractArray{<:AbstractDofMap})
  S = Snapshots
  N = ndims(s)
  block_cache = Array{S,N}(undef,size(s))
  return block_cache
end

function DofMaps.change_dof_map(s::BlockSnapshots,i::AbstractArray{<:AbstractDofMap})
  array = return_cache(change_dof_map,s,i)
  touched = s.touched
  for n in eachindex(touched)
    if touched[n]
      array[n] = change_dof_map(s[n],i[n])
    end
  end
  return BlockSnapshots(array,touched)
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

# linear algebra

function Base.:*(A::Snapshots{T,2},B::Snapshots{S,2}) where {T,S}
  consec_mul(get_indexed_data(A),get_indexed_data(B))
end

function Base.:*(A::Snapshots{T,2},B::Adjoint{S,<:Snapshots}) where {T,S}
  consec_mul(get_indexed_data(A),adjoint(get_indexed_data(B.parent)))
end

function Base.:*(A::Snapshots{T,2},B::AbstractMatrix{S}) where {T,S}
  consec_mul(get_indexed_data(A),B)
end

function Base.:*(A::Snapshots{T,2},B::Adjoint{T,<:AbstractMatrix{S}}) where {T,S}
  consec_mul(get_indexed_data(A),B)
end

function Base.:*(A::Adjoint{T,<:Snapshots{T,2}},B::Snapshots{S,2}) where {T,S}
  consec_mul(adjoint(get_indexed_data(A.parent)),get_indexed_data(B))
end

function Base.:*(A::AbstractMatrix{T},B::Snapshots{S,2}) where {T,S}
  consec_mul(A,get_indexed_data(B))
end

function Base.:*(A::Adjoint{T,<:AbstractMatrix},B::Snapshots{S,2}) where {T,S}
  consec_mul(A,get_indexed_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::Snapshots{S,2}) where {T,S}
  consec_mul!(C,get_indexed_data(A),get_indexed_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::Adjoint{S,<:Snapshots}) where {T,S}
  consec_mul!(C,get_indexed_data(A),adjoint(get_indexed_data(B.parent)))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::AbstractMatrix{S}) where {T,S}
  consec_mul!(C,get_indexed_data(A),B)
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::Adjoint{T,<:AbstractMatrix{S}}) where {T,S}
  consec_mul!(C,get_indexed_data(A),B)
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Adjoint{T,<:Snapshots{T,2}},B::Snapshots{S,2}) where {T,S}
  consec_mul!(C,adjoint(get_indexed_data(A.parent)),get_indexed_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::AbstractMatrix{T},B::Snapshots{S,2}) where {T,S}
  consec_mul!(C,A,get_indexed_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Adjoint{T,<:AbstractMatrix},B::Snapshots{S,2}) where {T,S}
  consec_mul!(C,A,get_indexed_data(B))
end

consec_mul(A::AbstractArray,B::AbstractArray) = A*B
consec_mul!(C::AbstractArray,A::AbstractArray,B::AbstractArray) = mul!(C,A,B)

for T in (:ConsecutiveParamArray,:ConsecutiveParamSparseMatrix)
  @eval begin
    consec_mul(A::$T,B::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}}) where S = get_all_data(A)*B
    consec_mul(A::Adjoint{S,<:$T},B::Union{<:AbstractArray,Adjoint{U,<:AbstractArray}}) where {S,U} = adjoint(get_all_data(A.parent))*B
    consec_mul(A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::$T) where S = A*get_all_data(B)
    consec_mul(A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::Adjoint{U,<:$T}) where {S,U} = A*adjoint(get_all_data(B.parent))
    consec_mul!(C::AbstractArray,A::$T,B::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}}) where S = mul!(C,get_all_data(A),B)
    consec_mul!(C::AbstractArray,A::Adjoint{S,<:$T},B::Union{<:AbstractArray,Adjoint{U,<:AbstractArray}}) where {S,U} = mul!(C,adjoint(get_all_data(A.parent)),B)
    consec_mul!(C::AbstractArray,A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::$T) where S = mul!(C,A,get_all_data(B))
    consec_mul!(C::AbstractArray,A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::Adjoint{U,<:$T}) where {S,U} = mul!(C,A,adjoint(get_all_data(B.parent)))
  end
end
