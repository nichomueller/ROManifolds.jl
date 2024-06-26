"""
    abstract type AbstractTransientSnapshots{T,N,L,D,I,R<:TransientParamRealization}
      <: AbstractSnapshots{T,N,L,D,I,R} end

Transient specialization of an [`AbstractSnapshots`](@ref). The dimension `N` of a
AbstractSteadySnapshots is equal to `D` + 2, where `D` represents the number of
spatial axes, to which a temporal and a parametric dimension are added.

Subtypes:
- [`TransientBasicSnapshots`](@ref)
- [`BasicSnapshots`](@ref)
- [`TransientSnapshotsAtIndices`](@ref)
- [`ModeTransientSnapshots`](@ref)

# Examples

```jldoctest
julia> ns1,ns2,nt,np = 2,2,1,2
(2, 2, 1, 2)
julia> data = [rand(ns1*ns2) for ip = 1:np*nt]
2-element Vector{Vector{Float64}}:
 [0.4684452123483283, 0.1195886171030737, 0.1151790990455997, 0.0375575515915656]
 [0.9095165124078269, 0.7346081836882059, 0.8939511550403715, 0.2288086807377305]
julia> i = IndexMap(collect(LinearIndices((ns1,ns2))))
2×2 IndexMap{2, Int64}:
 1  3
 2  4
julia> ptspace = TransientParamSpace(fill([0,1],3))
Set of tuples (p,t) in [[0, 1], [0, 1], [0, 1]] × 0:1
julia> r = realization(ptspace,nparams=np)
GenericTransientParamRealization{ParamRealization{Vector{Vector{Float64}}},
  Int64, Vector{Int64}}([
  [0.4021870679335007, 0.6585653527784044, 0.5110768420820191],
  [0.0950901750101361, 0.7049711670440882, 0.3490097863258958]],
  [1],
  0)
julia> s = Snapshots(ParamArray(data),i,r)
2×2×1×2 TransientBasicSnapshots{Float64, 4, 2, 2, IndexMap{2, Int64},
  GenericTransientParamRealization{ParamRealization{Vector{Vector{Float64}}}, Int64, Vector{Int64}},
  VectorOfVectors{Float64, 2}}:
  [:, :, 1, 1] =
  0.468445  0.115179
  0.119589  0.0375576

  [:, :, 1, 2] =
  0.909517  0.893951
  0.734608  0.228809
```

"""
abstract type AbstractTransientSnapshots{T,N,L,D,I,R<:TransientParamRealization} <: AbstractSnapshots{T,N,L,D,I,R} end

ParamDataStructures.num_times(s::AbstractTransientSnapshots) = num_times(get_realization(s))

Base.size(s::AbstractTransientSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))

"""
    struct TransientBasicSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R} end

Most standard implementation of a AbstractTransientSnapshots

"""
struct TransientBasicSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R}
  data::A
  index_map::I
  realization::R
  function TransientBasicSnapshots(
    data::A,
    index_map::I,
    realization::R
    ) where {T,N,L,D,R,A<:AbstractParamArray{T,N,L},I<:AbstractIndexMap{D}}
    new{T,D+2,L,D,I,R,A}(data,index_map,realization)
  end
end

function RBSteady.Snapshots(s::AbstractParamArray,i::AbstractIndexMap,r::TransientParamRealization)
  TransientBasicSnapshots(s,i,r)
end

ParamDataStructures.get_values(s::TransientBasicSnapshots) = s.data
IndexMaps.get_index_map(s::TransientBasicSnapshots) = s.index_map
RBSteady.get_realization(s::TransientBasicSnapshots) = s.realization

Base.@propagate_inbounds function Base.getindex(
  s::TransientBasicSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data,iparam+(itime-1)*num_params(s))
  getindex(sparam,ispace′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientBasicSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data,iparam+(itime-1)*num_params(s))
  setindex!(sparam,v,ispace′)
end

"""
    struct TransientSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R} end

Stores a vector of parametric arrays, obtained e.g. from a time marching scheme.
The inner length of the data corresponds to the length of the parameters, while
the outer length corresponds to the length of the time stencil. A TransientSnapshots
is indexed exacly as a [`TransientBasicSnapshots`](@ref)

"""
struct TransientSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R}
  data::Vector{A}
  index_map::I
  realization::R
  function TransientSnapshots(
    data::Vector{A},
    index_map::I,
    realization::R
    ) where {T,N,Lp,D,R,A<:AbstractParamArray{T,N,Lp},I<:AbstractIndexMap{D}}

    Lt = length(data)
    L = Lp*Lt
    new{T,D+2,L,D,I,R,A}(data,index_map,realization)
  end
end

function RBSteady.Snapshots(s::Vector{<:AbstractParamArray},i::AbstractIndexMap,r::TransientParamRealization)
  TransientSnapshots(s,i,r)
end

function ParamDataStructures.get_values(s::TransientSnapshots)
  item = testitem(first(s.data).data)
  data = array_of_similar_arrays(item,num_params(s)*num_times(s))
  @inbounds for (ipt,index) in enumerate(CartesianIndices((num_params(s),num_times(s))))
    ip,it = index.I
    @views data[ipt] = s.data[it][ip]
  end
  return data
end

IndexMaps.get_index_map(s::TransientSnapshots) = s.index_map
RBSteady.get_realization(s::TransientSnapshots) = s.realization

Base.@propagate_inbounds function Base.getindex(
  s::TransientSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data[itime],iparam)
  getindex(sparam,ispace′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data[itime],iparam)
  setindex!(sparam,v,ispace′)
end

"""
    struct TransientSnapshotsAtIndices{T,N,L,D,I,R,A<:AbstractTransientSnapshots{T,N,L,D,I,R},B,C
      } <: AbstractTransientSnapshots{T,N,L,D,I,R}

Represents a AbstractTransientSnapshots `snaps` whose parametric and temporal ranges
are restricted to the indices in `prange` and `trange`. This type essentially acts
as a view for suptypes of AbstractTransientSnapshots, at every space location, on
a selected number of parameter/time indices. An instance of TransientSnapshotsAtIndices
is created by calling the function [`select_snapshots`](@ref)

"""
struct TransientSnapshotsAtIndices{T,N,L,D,I,R,A<:AbstractTransientSnapshots{T,N,L,D,I,R},B,C} <: AbstractTransientSnapshots{T,N,L,D,I,R}
  snaps::A
  trange::B
  prange::C
end

function RBSteady.Snapshots(s::TransientSnapshotsAtIndices,i::AbstractIndexMap,r::TransientParamRealization)
  snaps = Snapshots(s.snaps,i,r)
  TransientSnapshotsAtIndices(snaps,s.trange,s.prange)
end

time_indices(s::TransientSnapshotsAtIndices) = s.trange
ParamDataStructures.num_times(s::TransientSnapshotsAtIndices) = length(time_indices(s))
RBSteady.param_indices(s::TransientSnapshotsAtIndices) = s.prange
ParamDataStructures.num_params(s::TransientSnapshotsAtIndices) = length(RBSteady.param_indices(s))

IndexMaps.get_index_map(s::TransientSnapshotsAtIndices) = get_index_map(s.snaps)
RBSteady.get_realization(s::TransientSnapshotsAtIndices) = get_realization(s.snaps)[s.prange,s.trange]

function ParamDataStructures.get_values(
  s::TransientSnapshotsAtIndices{T,N,L,D,I,R,<:TransientBasicSnapshots}
  ) where {T,N,L,D,I,R}

  snaps = s.snaps
  item = testitem(snaps.data)
  data = array_of_similar_arrays(item,num_params(s)*num_times(s))
  indices = CartesianIndices((RBSteady.param_indices(s),time_indices(s)))
  @inbounds for (ipt,index) in enumerate(indices)
    ip,it = index.I
    @views data[ipt] = snaps.data[ip+(it-1)*num_params(s)]
  end
  return data
end

function ParamDataStructures.get_values(
  s::TransientSnapshotsAtIndices{T,N,L,D,I,R,<:TransientSnapshots}
  ) where {T,N,L,D,I,R}

  snaps = s.snaps
  item = testitem(first(snaps.data).data)
  data = array_of_similar_arrays(item,num_params(s)*num_times(s))
  indices = CartesianIndices((RBSteady.param_indices(s),time_indices(s)))
  @inbounds for (ipt,index) in enumerate(indices)
    ip,it = index.I
    data[ipt] = snaps.data[it][ip]
  end
  return data
end

Base.@propagate_inbounds function Base.getindex(
  s::TransientSnapshotsAtIndices{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  itime′ = getindex(time_indices(s),itime)
  iparam′ = getindex(RBSteady.param_indices(s),iparam)
  getindex(s.snaps,ispace...,itime′,iparam′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientSnapshotsAtIndices{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  itime′ = getindex(time_indices(s),itime)
  iparam′ = getindex(RBSteady.param_indices(s),iparam)
  setindex!(s.snaps,v,ispace,itime′,iparam′)
end

function RBSteady.flatten_snapshots(s::Union{TransientSnapshots,TransientSnapshotsAtIndices})
  data = get_values(s)
  sbasic = Snapshots(data,get_index_map(s),get_realization(s))
  flatten_snapshots(sbasic)
end

function RBSteady.select_snapshots(s::TransientSnapshotsAtIndices,trange,prange)
  old_trange,old_prange = s.indices
  @check intersect(old_trange,trange) == trange
  @check intersect(old_prange,prange) == prange
  TransientSnapshotsAtIndices(s.snaps,trange,prange)
end

function RBSteady.select_snapshots(s::AbstractTransientSnapshots,trange,prange)
  trange = RBSteady.format_range(trange,num_times(s))
  prange = RBSteady.format_range(prange,num_params(s))
  TransientSnapshotsAtIndices(s,trange,prange)
end

function RBSteady.select_snapshots(s::AbstractTransientSnapshots,prange;trange=Base.OneTo(num_times(s)))
  select_snapshots(s,trange,prange)
end

function RBSteady.select_snapshots_entries(s::AbstractTransientSnapshots,srange,trange)
  _getindex(s::TransientBasicSnapshots,is,it,ip) = param_getindex(s.data,ip+(it-1)*num_params(s))[is]
  _getindex(s::TransientSnapshots,is,it,ip) = param_getindex(s.data[it],ip)[is]

  @assert length(srange) == length(trange)

  T = eltype(s)
  nval = length(srange)
  np = num_params(s)
  entries = array_of_similar_arrays(zeros(T,nval),np)

  @inbounds for ip = 1:np
    vip = entries.data[ip]
    for (i,(is,it)) in enumerate(zip(srange,trange))
      vip[i] = _getindex(s,is,it,ip)
    end
  end

  return entries
end

const MultiValueTransientBasicSnapshots{T,N,L,D,I<:AbstractMultiValueIndexMap{D},R,A} = TransientBasicSnapshots{T,N,L,D,I,R,A}
const MultiValueTransientSnapshots{T,N,L,D,I<:AbstractMultiValueIndexMap{D},R,A} = TransientSnapshots{T,N,L,D,I,R,A}
const MultiValueTransientSnapshotsAtIndices{T,N,L,D,I<:AbstractMultiValueIndexMap{D},R,A,B} = TransientSnapshotsAtIndices{T,N,L,D,I,R,A,B}
const TransientMultiValueSnapshots{T,N,L,D,I<:AbstractMultiValueIndexMap{D},R,A} = Union{
  MultiValueTransientBasicSnapshots{T,N,L,D,I,R,A},
  MultiValueTransientSnapshots{T,N,L,D,I,R,A},
  MultiValueTransientSnapshotsAtIndices{T,N,L,D,I,R,A}
}

TensorValues.num_components(s::TransientMultiValueSnapshots) = num_components(get_index_map(i))

function IndexMaps.get_component(s::TransientMultiValueSnapshots,args...;kwargs...)
  i′ = get_component(get_index_map(s),args...;kwargs...)
  return Snapshots(get_values(s),i′,get_realization(s))
end

function IndexMaps.split_components(s::TransientMultiValueSnapshots)
  i′ = split_components(get_index_map(s))
  return Snapshots(get_values(s),i′,get_realization(s))
end

function IndexMaps.merge_components(s::TransientMultiValueSnapshots)
  i′ = merge_components(get_index_map(s))
  return Snapshots(get_values(s),i′,get_realization(s))
end

const TransientSparseSnapshots{T,N,L,D,I,R,A<:MatrixOfSparseMatricesCSC} = Union{
  TransientBasicSnapshots{T,N,L,D,I,R,A},
  TransientSnapshots{T,N,L,D,I,R,A},
  TransientMultiValueSnapshots{T,N,L,D,I,R,A}
}

function ParamDataStructures.recast(s::TransientSparseSnapshots,a::AbstractVector{<:AbstractArray{T,3}}) where T
  index_map = get_index_map(s)
  ls = IndexMaps.get_univariate_sparsity(index_map)
  asparse = map(SparseCore,a,ls)
  return asparse
end

"""
    const UnfoldingTransientSnapshots{T,L,I<:TrivialIndexMap,R}
      = AbstractTransientSnapshots{T,3,L,1,I,R}

"""
const UnfoldingTransientSnapshots{T,L,I<:TrivialIndexMap,R} = AbstractTransientSnapshots{T,3,L,1,I,R}
const UnfoldingTransientSparseSnapshots{T,L,I<:TrivialIndexMap,R,A<:MatrixOfSparseMatricesCSC} = TransientSparseSnapshots{T,3,L,1,I,R,A}

function ParamDataStructures.recast(s::UnfoldingTransientSparseSnapshots,a::AbstractMatrix)
  return recast(s.data,a)
end

abstract type ModeAxes end
struct Mode1Axes <: ModeAxes end
struct Mode2Axes <: ModeAxes end

change_mode(::Mode1Axes) = Mode2Axes()
change_mode(::Mode2Axes) = Mode1Axes()

"""
    struct ModeTransientSnapshots{M<:ModeAxes,T,L,I,R,A<:UnfoldingTransientSnapshots{T,L,I,R}}
      <: AbstractTransientSnapshots{T,2,L,1,I,R}

Represents a AbstractTransientSnapshots with a TrivialIndexMap indexing strategy
as an AbstractMatrix with a system of mode-unfolding representations. Only two
mode-unfolding representations are considered:

Mode1Axes:

[u(x1,t1,μ1) ⋯ u(x1,t1,μP) u(x1,t2,μ1) ⋯ u(x1,t2,μP) u(x1,t3,μ1) ⋯ ⋯ u(x1,tT,μ1) ⋯ u(x1,tT,μP)]
      ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
 u(xN,t1,μ1) ⋯ u(xN,t1,μP) u(xN,t2,μ1) ⋯ u(xN,t2,μP) u(xN,t3,μ1) ⋯ ⋯ u(xN,tT,μ1) ⋯ u(xN,tT,μP)]

Mode2Axes:

[u(x1,t1,μ1) ⋯ u(x1,t1,μP) u(x2,t1,μ1) ⋯ u(x2,t1,μP) u(x3,t1,μ1) ⋯ ⋯ u(xN,t1,μ1) ⋯ u(xN,t1,μP)]
      ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
 u(x1,tT,μ1) ⋯ u(x1,tT,μP) u(x2,tT,μ1) ⋯ u(x2,tT,μP) u(x3,tT,μ1) ⋯ ⋯ u(xN,tT,μ1) ⋯ u(xN,tT,μP)]

"""
struct ModeTransientSnapshots{M<:ModeAxes,T,L,I,R,A<:UnfoldingTransientSnapshots{T,L,I,R}} <: AbstractTransientSnapshots{T,2,L,1,I,R}
  snaps::A
  mode::M
end

function ModeTransientSnapshots(s::AbstractTransientSnapshots)
  ModeTransientSnapshots(s,get_mode(s))
end

function RBSteady.flatten_snapshots(s::AbstractTransientSnapshots)
  s′ = change_index_map(TrivialIndexMap,s)
  ModeTransientSnapshots(s′)
end

RBSteady.num_space_dofs(s::ModeTransientSnapshots) = prod(num_space_dofs(s.snaps))
ParamDataStructures.get_values(s::ModeTransientSnapshots) = get_values(s.snaps)
RBSteady.get_realization(s::ModeTransientSnapshots) = get_realization(s.snaps)

change_mode(s::UnfoldingTransientSnapshots) = ModeTransientSnapshots(s,change_mode(get_mode(s)))
change_mode(s::ModeTransientSnapshots) = ModeTransientSnapshots(s.snaps,change_mode(get_mode(s)))

get_mode(s::UnfoldingTransientSnapshots) = Mode1Axes()
get_mode(s::ModeTransientSnapshots) = s.mode

function RBSteady.select_snapshots_entries(s::UnfoldingTransientSnapshots,srange,trange)
  _getindex(s::TransientBasicSnapshots,is,it,ip) = param_getindex(s.data,ip+(it-1)*num_params(s))[is]
  _getindex(s::TransientSnapshots,is,it,ip) = param_getindex(s.data[it],ip)[is]

  T = eltype(s)
  nval = length(srange),length(trange)
  np = num_params(s)
  entries = array_of_similar_arrays(zeros(T,nval),np)

  @inbounds for ip = 1:np, (i,it) = enumerate(trange)
    entries.data[ip][:,i] = _getindex(s,srange,it,ip)
  end

  return entries
end

const Mode1TransientSnapshots{T,L,I,R,A} = ModeTransientSnapshots{Mode1Axes,T,L,I,R,A}
const Mode2TransientSnapshots{T,L,I,R,A} = ModeTransientSnapshots{Mode2Axes,T,L,I,R,A}

Base.size(s::Mode1TransientSnapshots) = (num_space_dofs(s),num_times(s)*num_params(s))
Base.size(s::Mode2TransientSnapshots) = (num_times(s),num_space_dofs(s)*num_params(s))

Base.@propagate_inbounds function Base.getindex(s::Mode1TransientSnapshots,ispace::Integer,icol::Integer)
  @boundscheck checkbounds(s,ispace,icol)
  itime = slow_index(icol,num_params(s))
  iparam = fast_index(icol,num_params(s))
  getindex(s.snaps,ispace,itime,iparam)
end

Base.@propagate_inbounds function Base.setindex!(s::Mode1TransientSnapshots,v,ispace::Integer,icol::Integer)
  @boundscheck checkbounds(s,ispace,icol)
  itime = slow_index(icol,num_params(s))
  iparam = fast_index(icol,num_params(s))
  setindex!(s.snaps,v,ispace,itime,iparam)
end

Base.@propagate_inbounds function Base.getindex(s::Mode2TransientSnapshots,itime::Integer,icol::Integer)
  @boundscheck checkbounds(s,itime,icol)
  ispace = slow_index(icol,num_params(s))
  iparam = fast_index(icol,num_params(s))
  getindex(s.snaps,ispace,itime,iparam)
end

Base.@propagate_inbounds function Base.setindex!(s::Mode2TransientSnapshots,v,itime::Integer,icol::Integer)
  @boundscheck checkbounds(s,itime,icol)
  ispace = slow_index(icol,num_params(s))
  iparam = fast_index(icol,num_params(s))
  setindex!(s.snaps,v,ispace,itime,iparam)
end

# compression operation

_compress(s,a,X::AbstractMatrix) = a'*X*s
_compress(s,a,args...) = a'*s

function compress(s::Mode1TransientSnapshots,a::AbstractMatrix,args...;swap_mode=true)
  s′ = _compress(collect(s),a,args...)
  if swap_mode
    s′ = change_mode(s′,num_params(s))
  end
  return s′
end

function change_mode(a::AbstractMatrix,np::Integer)
  n1 = size(a,1)
  n2 = Int(size(a,2)/np)
  a′ = zeros(eltype(a),n2,n1*np)
  @inbounds for i = 1:np
    @views a′[:,(i-1)*n1+1:i*n1] = a[:,i:np:np*n2]'
  end
  return a′
end

function Base.:*(
  A::Adjoint{T,<:Mode1TransientSnapshots{T,L,I,R,<:TransientBasicSnapshots}},
  B::Mode1TransientSnapshots{T′,L′,I′,R′,<:TransientBasicSnapshots}
  ) where {T,L,I,R,T′,L′,I′,R′}

  a = A.parent.snaps.data
  b = B.snaps.data
  nsA = num_space_dofs(A.parent)
  nsB = num_space_dofs(B)
  @check nsA == nsB
  Tab = promote_eltype(T,T′)
  c = zeros(Tab,(size(a,2),size(B,2)))

  @inbounds for iA in axes(a,2)
    row = a[iA]
    @inbounds for iB in axes(B,2)
      col = b[iB]
      c[iA,iB] = dot(row,col)
    end
  end

  return c
end

function Base.:*(
  A::Adjoint{T,<:AbstractMatrix},
  B::Mode1TransientSnapshots{T′,L′,I′,R′,<:TransientBasicSnapshots}
  ) where {T,T′,L′,I′,R′}

  a = A.parent
  b = B.snaps.data
  @check size(a,1) == nsB
  Tab = promote_eltype(T,T′)
  c = zeros(Tab,(size(a,2),size(B,2)))

  @inbounds for iB in axes(B,2)
    col = b[iB]
    @inbounds for iA in axes(a,2)
      @fastmath iB = (itB-1)*np + ipB
      row = a[:,iA]
      c[iA,iB] = dot(row,col)
    end
  end

  return c
end

function Base.:*(
  A::Adjoint{T,<:Mode1TransientSnapshots{T,L,I,R,<:TransientSnapshots}},
  B::Mode1TransientSnapshots{T′,L′,I′,R′,<:TransientSnapshots}
  ) where {T,L,I,R,T′,L′,I′,R′}

  a = A.parent.snaps.data
  b = B.snaps.data
  nsA,ntA,npA = num_space_dofs(A.parent),num_times(A.parent),num_params(A.parent)
  nsB,ntB,npB = num_space_dofs(B),num_times(B),num_params(B)
  @check nsA == nsB
  Tab = promote_eltype(T,T′)
  c = zeros(Tab,(ntA*npA,ntB*npB))

  @inbounds for itA in 1:ntA
    row_block = a[itA]
    @inbounds for itB in 1:ntB
      col_block = b[itB]
      @inbounds for ipA in 1:npA
        @fastmath iA = (itA-1)*npA + ipA
        row = row_block[ipA]
        @inbounds for ipB in 1:npB
          @fastmath iB = (itB-1)*npB + ipB
          col = col_block[ipB]
          c[iA,iB] = dot(row,col)
        end
      end
    end
  end

  return c
end

function Base.:*(
  A::Adjoint{T,<:AbstractMatrix},
  B::Mode1TransientSnapshots{T′,L′,I′,R′,<:TransientSnapshots}
  ) where {T,T′,L′,I′,R′}

  a = A.parent
  b = B.snaps.data
  ns,nt,np = num_space_dofs(B),num_times(B),num_params(B)
  @check size(a,1) == ns
  Tab = promote_eltype(T,T′)
  c = zeros(Tab,(size(a,2),nt*np))

  @inbounds for itB in 1:nt
    col_block = b[itB]
    @inbounds for ipB in 1:np
      @fastmath iB = (itB-1)*np + ipB
      col = col_block[ipB]
      @inbounds for iA in axes(a,2)
        row = a[:,iA]
        c[iA,iB] = dot(row,col)
      end
    end
  end

  return c
end

# block snapshots

function RBSteady.Snapshots(
  data::AbstractVector{<:BlockArrayOfArrays},
  i::AbstractVector{<:AbstractIndexMap},
  r::AbstractParamRealization)

  block_values = blocks.(data)
  nblocks = blocksize(first(data))
  active_block_ids = findall(!iszero,blocks(first(data)))
  block_map = BlockMap(nblocks,active_block_ids)
  active_block_snaps = [Snapshots(map(v->getindex(v,n),block_values),i[n],r) for n in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps)
end

function RBSteady.select_snapshots_entries(
  s::BlockSnapshots{S,N},
  srange::ArrayBlock{<:Any,N},
  trange::ArrayBlock{<:Any,N}) where {S,N}

  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  active_block_snaps = [select_snapshots_entries(s[n],srange[n],trange[n]) for n in active_block_ids]
  return_cache(block_map,active_block_snaps...)
end
