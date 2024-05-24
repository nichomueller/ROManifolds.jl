abstract type TTSnapshots{T,N} <: AbstractSnapshots{T,N} end

Base.size(s::TTSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))
Base.length(s::TTSnapshots) = prod(size(s))
Base.axes(s::TTSnapshots) = Base.OneTo.(size(s))

num_space_dofs(s::TTSnapshots) = size(get_index_map(s))

Base.IndexStyle(::Type{<:TTSnapshots}) = IndexCartesian()

function Base.getindex(s::TTSnapshots,i...)
  view(s,i...)
end

function Base.getindex(s::TTSnapshots,i::Integer...)
  getindex(s,CartesianIndex(i))
end

function Base.getindex(s::TTSnapshots,i::Integer)
  nspace = prod(num_space_dofs(s))
  ispace = tensorize_indices(i,num_space_dofs(s))
  itime = fast_index(i,nspace*num_times(s))
  iparam = slow_index(i,nspace*num_times(s))
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.getindex(s::TTSnapshots{T,N},i::CartesianIndex{N}) where {T,N}
  ispace = CartesianIndex(i.I[1:N-2])
  itime = i.I[end-1]
  iparam = i.I[end]
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.setindex!(s::TTSnapshots{T,N},v,i::Integer...) where {T,N}
  ispace = CartesianIndex(i[1:N-2])
  itime = i[end-1]
  iparam = i[end]
  tensor_setindex!(s,v,ispace,itime,iparam)
end

reverse_snapshots(s::TTSnapshots) = s

#= representation of a standard tensor-train snapshot
   [ [u(x1,t1,μ1) ⋯ u(x1,t1,μP)] [u(x1,t2,μ1) ⋯ u(x1,t2,μP)] [u(x1,t3,μ1) ⋯] [⋯] [u(x1,tT,μ1) ⋯ u(x1,tT,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(xN,t1,μ1) ⋯ u(xN,t1,μP)] [u(xN,t2,μ1) ⋯ u(xN,t2,μP)] [u(xN,t3,μ1) ⋯] [⋯] [u(xN,tT,μ1) ⋯ u(xN,tT,μP)] ]
=#

struct BasicTTSnapshots{T,N,P,R} <: TTSnapshots{T,N}
  values::P
  realization::R
  function BasicTTSnapshots(values::P,realization::R) where {P<:ParamTTArray,R}
    T = eltype(P)
    D = ndims(get_index_map(values))
    N = D+2
    new{T,N,P,R}(values,realization)
  end
end

function BasicSnapshots(values::ParamTTArray,realization::TransientParamRealization)
  BasicTTSnapshots(values,realization)
end

function BasicSnapshots(s::BasicTTSnapshots)
  s
end

FEM.get_index_map(s::BasicTTSnapshots) = get_index_map(s.values)

function tensor_getindex(s::BasicTTSnapshots,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  s.values[iparam+(itime-1)*num_params(s)][perm_ispace]
end

function tensor_setindex!(s::BasicTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  s.values[iparam+(itime-1)*num_params(s)][perm_ispace] = v
end

struct TransientTTSnapshots{T,N,P,R,V} <: TTSnapshots{T,N}
  values::V
  realization::R
  function TransientTTSnapshots(
    values::AbstractVector{P},
    realization::R
    ) where {P<:ParamTTArray,R<:TransientParamRealization}

    V = typeof(values)
    T = eltype(P)
    D = ndims(get_index_map(first(values)))
    N = D+2
    new{T,N,P,R,V}(values,realization)
  end
end

function TransientSnapshots(
  values::AbstractVector{P},
  realization::TransientParamRealization
  ) where P<:ParamTTArray

  TransientTTSnapshots(values,realization)
end

FEM.get_index_map(s::TransientTTSnapshots) = get_index_map(first(s.values))

function tensor_getindex(s::TransientTTSnapshots,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  s.values[itime][iparam][perm_ispace]
end

function tensor_setindex!(s::TransientTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  s.values[itime][iparam][perm_ispace] = v
end

function BasicSnapshots(
  s::TransientTTSnapshots{T,N,<:ParamArray{T,M,L,A}}
  ) where {T,N,M,L,A}

  nt = num_times(s)
  np = num_params(s)
  array = Vector{eltype(A)}(undef,nt*np)
  @inbounds for i = 1:nt*np
    it = slow_index(i,np)
    ip = fast_index(i,np)
    array[i] = s.values[it][ip]
  end
  basic_values = ParamArray(array)
  BasicSnapshots(basic_values,s.realization)
end

function FEM.get_values(s::TransientTTSnapshots)
  get_values(BasicSnapshots(s))
end

struct SelectedTTSnapshotsAtIndices{T,N,S,I} <: TTSnapshots{T,N}
  snaps::S
  selected_indices::I
  function SelectedTTSnapshotsAtIndices(
    snaps::TTSnapshots{T,N},
    selected_indices::I
    ) where {T,N,I}

    S = typeof(snaps)
    new{T,N,S,I}(snaps,selected_indices)
  end
end

function SelectedSnapshotsAtIndices(snaps::TTSnapshots,args...)
  SelectedTTSnapshotsAtIndices(snaps,args...)
end

function SelectedSnapshotsAtIndices(s::SelectedTTSnapshotsAtIndices,selected_indices)
  new_srange,new_trange,new_prange = selected_indices
  old_srange,old_trange,old_prange = s.selected_indices
  @check intersect(old_srange,new_srange) == new_srange
  @check intersect(old_trange,new_trange) == new_trange
  @check intersect(old_prange,new_prange) == new_prange
  SelectedTTSnapshotsAtIndices(s.snaps,selected_indices)
end

function select_snapshots(s::TTSnapshots,spacerange,timerange,paramrange)
  @check isa(spacerange,Union{Colon,CartesianIndices})
  srange = isa(spacerange,Colon) ? CartesianIndices(num_space_dofs(s)) : spacerange
  trange = isa(timerange,Colon) ? LinearIndices((num_times(s),)) : timerange
  trange = isa(trange,Integer) ? [trange] : trange
  prange = isa(paramrange,Colon) ? LinearIndices((num_params(s),)) : paramrange
  prange = isa(prange,Integer) ? [prange] : prange
  selected_indices = (srange,trange,prange)
  SelectedSnapshotsAtIndices(s,selected_indices)
end

space_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[1]
time_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[2]
param_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[3]
num_space_dofs(s::SelectedTTSnapshotsAtIndices) = size(space_indices(s))
FEM.num_times(s::SelectedTTSnapshotsAtIndices) = length(time_indices(s))
FEM.num_params(s::SelectedTTSnapshotsAtIndices) = length(param_indices(s))

FEM.get_index_map(s::SelectedTTSnapshotsAtIndices) = get_index_map(s.snaps)

function tensor_getindex(s::SelectedTTSnapshotsAtIndices,ispace,itime,iparam)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_getindex(s.snaps,is,it,ip)
end

function tensor_setindex!(s::SelectedTTSnapshotsAtIndices,v,ispace,itime,iparam)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedTTSnapshotsAtIndices{T,N,<:BasicTTSnapshots}) where {T,N}
  v = get_values(s.snaps)
  array = Vector{typeof(first(v))}(undef,num_times(s)*num_params(s))
  ispace = space_indices(s)
  index_map = get_index_map(s)
  perm_ispace = map(i->index_map[CartesianIndex(i)],ispace)
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      array[(i-1)*num_params(s)+j] = v[(it-1)*num_params(s)+jp][perm_ispace]
    end
  end
  ParamArray(array)
end

function FEM.get_values(s::SelectedTTSnapshotsAtIndices{T,N,<:TransientTTSnapshots}) where {T,N}
  get_values(BasicSnapshots(s))
end

function get_realization(s::SelectedTTSnapshotsAtIndices)
  r = get_realization(s.snaps)
  r[param_indices(s),time_indices(s)]
end

function BasicSnapshots(s::SelectedTTSnapshotsAtIndices{T,N,<:BasicTTSnapshots}) where {T,N}
  values = get_values(s)
  r = get_realization(s)
  BasicTTSnapshots(values,r)
end

function _to_linear_indices(i::CartesianIndices,args...)
  vec(LinearIndices(i))
end
function _to_linear_indices(i::Tuple{Vararg{AbstractVector}},sizes)
  for k = 1:length(i)-1
    i[k+1] .+= sizes[k]
  end
  vcat(i...)
end

function BasicSnapshots(s::SelectedTTSnapshotsAtIndices{T,N,<:TransientTTSnapshots}) where {T,N}
  v = s.snaps.values
  basic_values = Vector{typeof(first(first(v)))}(undef,num_times(s)*num_params(s))
  ispace = space_indices(s)
  lispace = _to_linear_indices(ispace,num_space_dofs(s.snaps))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      basic_values[(i-1)*num_params(s)+j] = vec(v[it][jp][lispace])
    end
  end
  r = get_realization(s)
  BasicTTSnapshots(ParamArray(basic_values),r)
end

function select_snapshots_entries(s::TTSnapshots,ispace,itime)
  cids = CartesianIndices(num_space_dofs(s)) # is this ok?
  cispace = map(i->getindex(cids,i),ispace)
  _select_snapshots_entries(s,cispace,itime)
end

function _select_snapshots_entries(s::TTSnapshots{T},ispace,itime) where T
  @assert length(ispace) == length(itime)
  nval = length(ispace)
  np = num_params(s)
  values = allocate_param_array(zeros(T,nval),np)
  for ip = 1:np
    vip = values[ip]
    for (istp,(is,it)) in enumerate(zip(ispace,itime))
      vip[istp] = tensor_getindex(s,is,it,ip)
    end
  end
  return values
end

function vectorize_index_map(s::TTSnapshots)
  i = get_index_map(s)
  vi = FEM.vectorize_index_map(i)
  VectorizedTTSnapshots(s,vi)
end

struct VectorizedTTSnapshots{T,S,I} <: TTSnapshots{T,3}
  snaps::S
  index_map::I
  function VectorizedTTSnapshots(snaps::S,index_map::I) where {T,S<:TTSnapshots{T},I}
    new{T,S,I}(snaps,index_map)
  end
end

FEM.get_index_map(s::VectorizedTTSnapshots) = s.index_map

function get_realization(s::VectorizedTTSnapshots)
  get_realization(s.snaps)
end

function tensor_getindex(s::VectorizedTTSnapshots{T,<:BasicTTSnapshots},ispace,itime,iparam) where T
  perm_ispace = get_index_map(s)[ispace]
  s.snaps.values[iparam+(itime-1)*num_params(s)][perm_ispace]
end

function tensor_getindex(s::VectorizedTTSnapshots{T,<:TransientTTSnapshots},ispace,itime,iparam) where T
  perm_ispace = get_index_map(s)[ispace]
  s.snaps.values[itime][iparam][perm_ispace]
end

FEM.num_times(s::VectorizedTTSnapshots{T,<:SelectedTTSnapshotsAtIndices}) where T = num_times(s.snaps)
FEM.num_params(s::VectorizedTTSnapshots{T,<:SelectedTTSnapshotsAtIndices}) where T = num_params(s.snaps)

function tensor_getindex(s::VectorizedTTSnapshots{T,<:SelectedTTSnapshotsAtIndices},ispace,itime,iparam) where T
  is = ispace
  it = time_indices(s.snaps)[itime]
  ip = param_indices(s.snaps)[iparam]
  tensor_getindex(VectorizedTTSnapshots(s.snaps.snaps,s.index_map),is,it,ip)
end

const BasicNnzTTSnapshots = BasicTTSnapshots{T,N,P,R} where {T,N,P<:FEM.ParamTTSparseMatrix,R}
const TransientNnzTTSnapshots = TransientTTSnapshots{T,N,P,R,V} where {T,N,P<:FEM.ParamTTSparseMatrix,R,V}
const SelectedNnzTTSnapshotsAtIndices = SelectedTTSnapshotsAtIndices{T,N,S,I} where {T,N,S<:Union{BasicNnzTTSnapshots,TransientNnzTTSnapshots},I}
const NnzTTSnapshots = Union{
  BasicNnzTTSnapshots{T,N},
  TransientNnzTTSnapshots{T,N},
  SelectedNnzTTSnapshotsAtIndices{T,N}} where {T,N}

function select_snapshots_entries(s::NnzTTSnapshots,ispace,itime)
  index_map = get_index_map(s)
  cispace = map(i->findfirst(index_map.==i),ispace)
  _select_snapshots_entries(s,cispace,itime)
end

function recast(s::NnzTTSnapshots,a::AbstractVector{<:AbstractArray{T,3}}) where T
  index_map = get_index_map(s)
  g2l = fill(FEM.get_global_2_local_map(index_map),length(a))
  ls = FEM.get_univariate_sparsity(index_map)
  asparse = map(SparseCore,a,g2l,ls)
  return asparse
end

abstract type SparseCore{T} <: AbstractArray{T,4} end

Base.IndexStyle(::Type{<:SparseCore}) = IndexCartesian()

struct SparseCoreCSC{T,Ti,D} <: SparseCore{T}
  array::Array{T,3}
  global_2_local::IndexMap{D}
  sparsity::SparsityPatternCSC{T,Ti}
  sparse_indexes::Vector{CartesianIndex{2}}
end

function SparseCore(
  array::AbstractArray{T},
  global_2_local::AbstractIndexMap{D},
  sparsity::SparsityPatternCSC{T}) where {T,D}

  irows,icols,_ = findnz(sparsity)
  SparseCoreCSC(array,global_2_local,sparsity,CartesianIndex.(irows,icols))
end

Base.size(a::SparseCoreCSC) = (size(a.array,1),FEM.num_rows(a.sparsity),
  FEM.num_cols(a.sparsity),size(a.array,3))

function Base.getindex(a::SparseCoreCSC,i::Integer...)
  getindex(a,CartesianIndex(i))
end

function Base.getindex(a::SparseCoreCSC,i::CartesianIndex{4})
  is_nnz = CartesianIndex(i.I[2:3]) ∈ a.sparse_indexes
  sparse_getindex(Val(is_nnz),a,i)
end

function sparse_getindex(::Val{false},a::SparseCoreCSC{T},i::CartesianIndex{4}) where T
  zero(T)
end

function sparse_getindex(::Val{true},a::SparseCoreCSC{T},i::CartesianIndex{4}) where T
  i2 = findfirst(a.sparse_indexes .== [CartesianIndex(i.I[2:3])])
  i1,i3 = i.I[1],i.I[4]
  getindex(a.array,CartesianIndex((i1,i2,i3)))
end

function _cores2basis(a::SparseCoreCSC{S},b::SparseCoreCSC{T}) where {S,T}
  @check size(a,4) == size(b,1)
  @check a.global_2_local == b.global_2_local
  # local_2_global = sortperm(a.global_2_local[:])
  g2l = vec(a.global_2_local)
  TS = promote_type(T,S)
  nrows = size(a,2)*size(b,2)
  ncols = size(a,3)*size(b,3)
  ab = zeros(TS,size(a,1),nrows*ncols,size(b,4))
  for i = axes(a,1), j = axes(b,4)
    for α = axes(a,4)
      @inbounds @views ab[i,g2l,j] += kronecker(b.array[α,:,j],a.array[i,:,α])
    end
  end
  return ab
end
