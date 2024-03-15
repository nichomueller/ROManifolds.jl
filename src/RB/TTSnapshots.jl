abstract type TTSnapshots{T,N} <: AbstractSnapshots{T,N} end

Base.size(s::TTSnapshots) = (num_space_dofs(s),num_times(s),num_params(s))
Base.length(s::TTSnapshots) = prod(size(s))
Base.axes(s::TTSnapshots) = Base.OneTo.(size(s))

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
    N = FEM.get_dim(P)+2
    new{T,N,P,R}(values,realization)
  end
end

function BasicSnapshots(values::ParamTTArray,realization::TransientParamRealization,args...)
  BasicTTSnapshots(values,realization)
end

function BasicSnapshots(s::BasicTTSnapshots)
  s
end

num_space_dofs(s::BasicTTSnapshots) = length(first(s.values))

function Base.getindex(s::BasicTTSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace]
end

function Base.setindex!(s::BasicTTSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace] = v
end

function to_standard_snapshots(s::BasicTTSnapshots)
  Snapshots(get_values(s.values),s.realization)
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
    N = FEM.get_dim(P)+2
    new{T,N,P,R,V}(values,realization)
  end
end

function TransientSnapshots(
  values::AbstractVector{P},realization::TransientParamRealization,args...) where P<:ParamTTArray
  TransientTTSnapshots(values,realization,args...)
end

num_space_dofs(s::TransientTTSnapshots) = length(first(first(s.values)))

function Base.getindex(s::TransientTTSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace]
end

function Base.setindex!(s::TransientTTSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace] = v
end

function BasicSnapshots(
  s::TransientTTSnapshots{T,<:ParamTTArray{T,N,A}}
  ) where {T,N,A}

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

function to_standard_snapshots(s::TransientTTSnapshots)
  Snapshots(map(get_values,s.values),s.realization)
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

space_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[1]
time_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[2]
param_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[3]
num_space_dofs(s::SelectedTTSnapshotsAtIndices) = length(space_indices(s))
FEM.num_times(s::SelectedTTSnapshotsAtIndices) = length(time_indices(s))
FEM.num_params(s::SelectedTTSnapshotsAtIndices) = length(param_indices(s))

function Base.getindex(
  s::SelectedTTSnapshotsAtIndices,
  ispace::Integer,itime::Integer,iparam::Integer)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  getindex(s.snaps,is,it,ip)
end

function Base.setindex!(
  s::SelectedTTSnapshotsAtIndices,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedTTSnapshotsAtIndices{T,N,<:BasicTTSnapshots}) where {T,N}
  @check space_indices(s) == Base.OneTo(num_space_dofs(s))
  v = get_values(s.snaps)
  array = Vector{typeof(first(v))}(undef,num_cols(s))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      array[(i-1)*num_params(s)+j] = v[(it-1)*num_params(s)+jp]
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

function BasicSnapshots(s::SelectedTTSnapshotsAtIndices{T,N,<:TransientTTSnapshots}) where {T,N}
  @check space_indices(s) == Base.OneTo(num_space_dofs(s))
  v = s.snaps.values
  basic_values = Vector{typeof(first(first(v)))}(undef,num_times(s)*num_params(s))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      basic_values[(i-1)*num_params(s)+j] = v[it][jp]
    end
  end
  r = get_realization(s)
  BasicTTSnapshots(ParamArray(basic_values),r)
end

function select_snapshots_entries(s::TTSnapshots,ispace,itime)
  _select_snapshots_entries(s,ispace,itime)
end

function _select_snapshots_entries(s::TTSnapshots{T},ispace,itime) where T
  @assert length(ispace) == length(itime)
  nval = length(ispace)
  np = num_params(s)
  values = allocate_param_array(zeros(T,nval),np)
  for ip = 1:np
    vip = values[ip]
    for (istp,(is,it)) in enumerate(zip(ispace,itime))
      vip[istp] = s[is,it,ip]
    end
  end
  return values
end

const BasicNnzTTSnapshots = BasicTTSnapshots{T,N,P,R} where {T,N,P<:ParamTTSparseMatrix,R}
const TransientNnzTTSnapshots = TransientTTSnapshots{T,N,P,R} where {T,N,P<:ParamTTSparseMatrix,R}
const SelectedNnzTTSnapshotsAtIndices = SelectedTTSnapshotsAtIndices{T,N,S,I} where {T,N,S<:Union{BasicNnzTTSnapshots,TransientNnzTTSnapshots},I}
const NnzTTSnapshots = Union{
  BasicNnzTTSnapshots{T,N},
  TransientNnzTTSnapshots{T,N},
  SelectedNnzTTSnapshotsAtIndices{T,N}} where {T,N}

num_space_dofs(s::BasicNnzTTSnapshots) = nnz(first(s.values))

function Base.getindex(s::BasicNnzTTSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[ispace]
end

function Base.setindex!(s::BasicNnzTTSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[ispace] = v
end

num_space_dofs(s::TransientNnzTTSnapshots) = nnz(first(first(s.values)))

function Base.getindex(
  s::TransientNnzTTSnapshots,
  ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[itime][iparam])[ispace]
end

function Base.setindex!(
  s::TransientNnzTTSnapshots,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[itime][iparam])[ispace] = v
end

sparsify_indices(s::BasicNnzTTSnapshots,srange::AbstractVector) = sparsify_indices(first(s.values),srange)
sparsify_indices(s::TransientNnzTTSnapshots,srange::AbstractVector) = sparsify_indices(first(first(s.values)),srange)

function select_snapshots_entries(s::NnzTTSnapshots,ispace,itime)
  _select_snapshots_entries(s,sparsify_indices(s,ispace),itime)
end

function get_nonzero_indices(s::NnzTTSnapshots)
  v = isa(s,BasicTTSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  return i .+ (j .- 1)*v.m
end

function recast(s::NnzTTSnapshots,a::AbstractMatrix)
  v = isa(s,BasicTTSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  m,n = size(v)
  nz = nnz(v)
  asparse = map(eachcol(a)) do v
    blocks = map(1:num_times(s)) do it
      sparse(i,j,v[(it-1)*nz+1:it*nz],m,n)
    end
    BDiagonal(blocks)
  end
  return VecOfBDiagonalSparseMat2Mat(asparse)
end

struct VecOfBDiagonalSparseMat2Mat{Tv,Ti,V} <: AbstractSparseMatrix{Tv,Ti}
  values::V
  function VecOfBDiagonalSparseMat2Mat(values::V
    ) where {Tv,Ti,V<:AbstractVector{<:BDiagonal{Tv,<:AbstractSparseMatrix{Tv,Ti}}}}

    new{Tv,Ti,V}(values)
  end
end

FEM.get_values(s::VecOfBDiagonalSparseMat2Mat) = s.values

num_space_dofs(s::VecOfBDiagonalSparseMat2Mat) = nnz(first(first(s.values).values))
FEM.num_times(s::VecOfBDiagonalSparseMat2Mat) = length(first(s.values).values)
Base.size(s::VecOfBDiagonalSparseMat2Mat) = (num_space_dofs(s)*num_times(s),length(s.values))

function Base.getindex(s::VecOfBDiagonalSparseMat2Mat,i::Integer,j::Integer)
  itime = slow_index(i,num_space_dofs(s))
  ispace = fast_index(i,num_space_dofs(s))
  nonzeros(s.values[j].values[itime])[ispace]
end

function Base.getindex(s::VecOfBDiagonalSparseMat2Mat,i,j)
  view(s,i,j)
end

function get_nonzero_indices(s::VecOfBDiagonalSparseMat2Mat)
  v = first(first(s.values).values)
  i,j, = findnz(v)
  return i .+ (j .- 1)*v.m
end
