"""
    abstract type AbstractSnapshots{T,N} <: AbstractParamData{T,N} end

Type representing N-dimensional arrays of snapshots. Subtypes must contain the
following information:

- data: a (parametric) array
- realization: a subtype of [`AbstractRealization`](@ref), representing the points
  in the parameter space used to compute the array `data`
- dof map: a subtype of [`AbstractDofMap`](@ref), representing a reindexing strategy
  for the array `data`

Subtypes:

- [`Snapshots`](@ref)
- [`BlockSnapshots`](@ref)
"""
abstract type AbstractSnapshots{T,N} <: AbstractParamData{T,N} end

"""
    get_realization(s::AbstractSnapshots) -> AbstractRealization

Returns the realizations associated to the snapshots `s`
"""
get_realization(s::AbstractSnapshots) = @abstractmethod

DofMaps.get_dof_map(s::AbstractSnapshots) = @abstractmethod

num_params(s::AbstractSnapshots) = num_params(get_realization(s))

"""
    abstract type Snapshots{T,N,I<:AbstractDofMap,R<:AbstractRealization,A}
      <: AbstractSnapshots{T,N} end

Type representing a collection of parametric abstract arrays of eltype `T`,
that are associated with a realization of type `R`. Unlike `AbstractParamArray`,
which are arrays of arrays, subtypes of `Snapshots` are arrays of numbers.

Subtypes:

- [`SteadySnapshots`](@ref)
- [`TransientSnapshots`](@ref)
"""
abstract type Snapshots{T,N,I<:AbstractDofMap,R<:AbstractRealization,A} <: AbstractSnapshots{T,N} end

function Snapshots(s::AbstractArray,i::AbstractDofMap,r::AbstractRealization)
  @abstractmethod
end

function Snapshots(s::AbstractParamArray,i::AbstractDofMap,r::AbstractRealization)
  Snapshots(get_all_data(s),i,r)
end

get_all_data(s::Snapshots) = @abstractmethod

get_param_data(s::Snapshots) = ConsecutiveParamArray(get_all_data(s))

num_space_dofs(s::Snapshots) = length(get_dof_map(s))

function Base.reshape(s::Snapshots,dims::Dims)
  reshape(get_all_data(s))
end

"""
    const SteadySnapshots{T,N,I,R<:Realization,A} = Snapshots{T,N,I,R,A}
"""
const SteadySnapshots{T,N,I,R<:Realization,A} = Snapshots{T,N,I,R,A}

"""
    space_dofs(s::SteadySnapshots{T,N}) where {T,N} -> NTuple{N-1,Integer}
    space_dofs(s::TransientSnapshots{T,N}) where {T,N} -> NTuple{N-2,Integer}

Returns the spatial size of the snapshots
"""
space_dofs(s::SteadySnapshots{T,N}) where {T,N} = size(get_all_data(s))[1:N-1]

Base.size(s::SteadySnapshots) = (space_dofs(s)...,num_params(s))

function select_snapshots(s::SteadySnapshots,prange)
  drange = view(get_all_data(s),:,prange)
  rrange = get_realization(s)[prange]
  Snapshots(drange,get_dof_map(s),rrange)
end

"""
    struct GenericSnapshots{T,N,I,R,A} <: Snapshots{T,N,I,R,A}
      data::A
      dof_map::I
      realization::R
    end

Most standard implementation of a [`Snapshots`](@ref)
"""
struct GenericSnapshots{T,N,I,R,A} <: Snapshots{T,N,I,R,A}
  data::A
  dof_map::I
  realization::R

  function GenericSnapshots(
    data::A,
    dof_map::I,
    realization::R
    ) where {T,N,R,A<:AbstractArray{T,N},I<:AbstractDofMap}

    new{T,N,I,R,A}(data,dof_map,realization)
  end
end

function Snapshots(s::AbstractParamArray,i::AbstractDofMap,r::Realization)
  GenericSnapshots(s,i,r)
end

get_all_data(s::GenericSnapshots) = s.data
DofMaps.get_dof_map(s::GenericSnapshots) = s.dof_map
get_realization(s::GenericSnapshots) = s.realization

function Base.getindex(s::GenericSnapshots{T,N},i::Vararg{Integer,N}) where {T,N}
  s.data[i...]
end

function Base.setindex!(s::GenericSnapshots{T,N},v,i::Vararg{Integer,N}) where {T,N}
  s.data[i...] = v
end

# sparse interface

"""
"""
const SparseSnapshots{T,N,I<:AbstractDofMap,R<:AbstractRealization,A<:ParamSparseMatrix} = Snapshots{T,N,I,R,A}

function Snapshots(s::ParamSparseMatrix,i::SparseMatrixDofMap,r::Realization)
  T = eltype(s)
  i = get_dof_map(s)
  data = get_all_data(s)
  idata = zeros(T,size(i)...,num_params(r))
  for ip in 1:num_params(r)
    for k in CartesianIndices(i)
      k′ = i[k]
      if k′ > 0
        idata[k.I...,ip] = data[k′,ip]
      end
    end
  end
  Snapshots(idata,i,r)
end

function get_param_data(s::SparseSnapshots)
  @notimplemented "We do not keep the parametric data of snapshots of sparse matrices"
end

function DofMaps.recast(a::AbstractArray,s::SparseSnapshots)
  return recast(a,get_dof_map(s))
end

# multi field interface

"""
    struct BlockSnapshots{S<:Snapshots,N} <: AbstractSnapshots{S,N}
      array::Array{S,N}
      touched::Array{Bool,N}
    end

Block container for Snapshots of type `S` in a `MultiField` setting. This
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

function get_param_data(s::BlockSnapshots)
  map(get_param_data,s.array) |> mortar
end

function Arrays.return_cache(::typeof(select_snapshots),s::BlockSnapshots,args...)
  S = Snapshots
  N = ndims(s)
  block_cache = Array{S,N}(undef,size(s))
  return block_cache
end

function select_snapshots(s::BlockSnapshots,args...)
  array = return_cache(select_snapshots,s,args...)
  touched = s.touched
  for i in eachindex(touched)
    if touched[i]
      array[i] = select_snapshots(s[i],args...)
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
  *(get_all_data(A),get_all_data(B))
end

function Base.:*(A::Snapshots{T,2},B::Adjoint{S,<:Snapshots}) where {T,S}
  *(get_all_data(A),adjoint(get_all_data(B.parent)))
end

function Base.:*(A::Snapshots{T,2},B::AbstractMatrix{S}) where {T,S}
  *(get_all_data(A),B)
end

function Base.:*(A::Snapshots{T,2},B::Adjoint{T,<:AbstractMatrix{S}}) where {T,S}
  *(get_all_data(A),B)
end

function Base.:*(A::Adjoint{T,<:Snapshots{T,2}},B::Snapshots{S,2}) where {T,S}
  *(adjoint(get_all_data(A.parent)),get_all_data(B))
end

function Base.:*(A::AbstractMatrix{T},B::Snapshots{S,2}) where {T,S}
  *(A,get_all_data(B))
end

function Base.:*(A::Adjoint{T,<:AbstractMatrix},B::Snapshots{S,2}) where {T,S}
  *(A,get_all_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::Snapshots{S,2}) where {T,S}
  mul!(C,get_all_data(A),get_all_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::Adjoint{S,<:Snapshots}) where {T,S}
  mul!(C,get_all_data(A),adjoint(get_all_data(B.parent)))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::AbstractMatrix{S}) where {T,S}
  mul!(C,get_all_data(A),B)
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::Adjoint{T,<:AbstractMatrix{S}}) where {T,S}
  mul!(C,get_all_data(A),B)
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Adjoint{T,<:Snapshots{T,2}},B::Snapshots{S,2}) where {T,S}
  mul!(C,adjoint(get_all_data(A.parent)),get_all_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::AbstractMatrix{T},B::Snapshots{S,2}) where {T,S}
  mul!(C,A,get_all_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Adjoint{T,<:AbstractMatrix},B::Snapshots{S,2}) where {T,S}
  mul!(C,A,get_all_data(B))
end
