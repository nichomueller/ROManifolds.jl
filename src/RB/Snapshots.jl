abstract type AbstractSnapshots{T,N} <: AbstractParamContainer{T,N} end

FEM.get_values(s::AbstractSnapshots) = copy(s.values)
get_realization(s::AbstractSnapshots) = copy(s.realization)
FEM.num_times(s::AbstractSnapshots) = num_times(get_realization(s))
FEM.num_params(s::AbstractSnapshots) = num_params(get_realization(s))
Base.eltype(::AbstractSnapshots{T}) where T = T
Base.eltype(::Type{<:AbstractSnapshots{T}}) where T = T
Base.ndims(::AbstractSnapshots{T,N}) where {T,N} = N
Base.ndims(::Type{<:AbstractSnapshots{T,N}}) where {T,N} = N

abstract type StandardSnapshots{M,T} <: AbstractSnapshots{T,2} end

#= mode-1 representation of a standard snapshot (default)
   [u(x1,t1,μ1) ⋯ u(x1,t1,μP),u(x1,t2,μ1),⋯,u(x1,t2,μP),u(x1,t3,μ1),⋯,⋯,u(x1,tT,μ1),⋯,u(x1,tT,μP)]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [u(xN,t1,μ1) ⋯ u(xN,t1,μP),u(xN,t2,μ1),⋯,u(xN,t2,μP),u(xN,t3,μ1),⋯,⋯,u(xN,tT,μ1),⋯,u(xN,tT,μP)]
=#
#= mode-2 representation of a standard snapshot
   [[u(x1,t1,μ1),⋯,u(x1,t1,μP),u(x2,t1,μ1),⋯,u(x2,t1,μP),u(x3,t1,μ1,⋯,⋯,u(xN,t1,μ1),⋯,u(xN,t1,μP)]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [[u(x1,tT,μ1),⋯,u(x1,tT,μP),u(x2,tT,μ1),⋯,u(x2,tT,μP),u(x3,tT,μ1,⋯,⋯,u(xN,tT,μ1),⋯,u(xN,tT,μP)]
=#

struct Mode1Axis end
struct Mode2Axis end

get_mode(s::StandardSnapshots) = s.mode
num_rows(s::StandardSnapshots{Mode1Axis}) = num_space_dofs(s)
num_rows(s::StandardSnapshots{Mode2Axis}) = num_times(s)
num_cols(s::StandardSnapshots{Mode1Axis}) = num_times(s)*num_params(s)
num_cols(s::StandardSnapshots{Mode2Axis}) = num_space_dofs(s)*num_params(s)

Base.length(s::StandardSnapshots) = num_rows(s)*num_cols(s)
Base.size(s::StandardSnapshots) = (num_rows(s),num_cols(s))
Base.axes(s::StandardSnapshots) = Base.OneTo.(size(s))

Base.IndexStyle(::Type{<:StandardSnapshots}) = IndexLinear()

function col_index(s::StandardSnapshots,mode2_index,param_index)
  (mode2_index .- 1)*num_params(s) .+ param_index
end
function col_index(s::StandardSnapshots,mode2_index::AbstractVector,param_index::AbstractVector)
  vec(transpose((mode2_index.-1)*num_params(s) .+ collect(param_index)'))
end
function col_index(s::StandardSnapshots{Mode1Axis},::Colon,param_index)
  col_index(s,1:num_times(s),param_index)
end
function col_index(s::StandardSnapshots{Mode2Axis},::Colon,param_index)
  col_index(s,1:num_space_dofs(s),param_index)
end
function col_index(s::StandardSnapshots{Mode1Axis},::Colon,::Colon)
  col_index(s,1:num_times(s),1:num_params(s))
end
function col_index(s::StandardSnapshots{Mode2Axis},::Colon,::Colon)
  col_index(s,1:num_space_dofs(s),1:num_params(s))
end

function Base.getindex(s::StandardSnapshots,i)
  nrow = num_rows(s)
  irow = fast_index(i,nrow)
  icol = slow_index(i,nrow)
  getindex(s,irow,icol)
end

function Base.getindex(s::StandardSnapshots,k::CartesianIndex)
  ispace,j = k.I
  getindex(s,ispace,j)
end

function Base.getindex(s::StandardSnapshots{Mode1Axis},ispace,j)
  np = num_params(s)
  itime = slow_index(j,np)
  iparam = fast_index(j,np)
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.getindex(s::StandardSnapshots{Mode2Axis},itime,j)
  np = num_params(s)
  ispace = slow_index(j,np)
  iparam = fast_index(j,np)
  tensor_getindex(s,ispace,itime,iparam)
end

function tensor_getindex(s::StandardSnapshots,ispace,itime,iparam)
  view(s,ispace,col_index(s,itime,iparam))
end

function Base.setindex!(s::StandardSnapshots,v,i)
  nrow = num_rows(s)
  irow = fast_index(i,nrow)
  icol = slow_index(i,nrow)
  setindex!(s,v,irow,icol)
end

function Base.setindex!(s::StandardSnapshots,v,k::CartesianIndex)
  ispace,j = k.I
  setindex!(s,v,ispace,j)
end

function Base.setindex!(s::StandardSnapshots{Mode1Axis},v,ispace,j)
  np = num_params(s)
  itime = slow_index(j,np)
  iparam = fast_index(j,np)
  tensor_setindex!(s,v,ispace,itime,iparam)
end

function Base.setindex!(s::StandardSnapshots{Mode2Axis},v,itime,j)
  np = num_params(s)
  ispace = slow_index(j,np)
  iparam = fast_index(j,np)
  tensor_setindex!(s,v,ispace,itime,iparam)
end

function tensor_setindex!(s::StandardSnapshots,v,ispace,itime,iparam)
  si = view(s,ispace,col_index(s,itime,iparam))
  si = v
end

_compress(s,a,X::AbstractMatrix) = a'*X*s
_compress(s,a,args...) = a'*s

# spatial compression: can input a norm matrix if needed
function compress(s::StandardSnapshots{Mode1Axis},a::AbstractMatrix,args...)
  compressed_values = _compress(s,a,args...)
  r = get_realization(s)
  Snapshots(compressed_values,r,Mode1Axis())
end

# temporal compression: it doesn't make sense to provide a norm matrix
function compress(s::StandardSnapshots{Mode2Axis},a::AbstractMatrix)
  compressed_values = _compress(s,a)
  r = get_realization(s)
  compressed_realization = r[:,axes(a,2)]
  Snapshots(compressed_values,compressed_realization,Mode2Axis())
end

function Snapshots(a::ArrayContribution,args...)
  contribution(a.trians) do trian
    Snapshots(a[trian],args...)
  end
end

function Snapshots(a::TupOfArrayContribution,args...)
  map(a->Snapshots(a,args...),a)
end

struct BasicSnapshots{M,T,P,R} <: StandardSnapshots{M,T}
  mode::M
  values::P
  realization::R
  function BasicSnapshots(
    values::P,
    realization::R,
    mode::M=Mode1Axis(),
    ) where {M,P<:AbstractParamContainer,R}

    T = eltype(P)
    new{M,T,P,R}(mode,values,realization)
  end
end

function BasicSnapshots(s::BasicSnapshots)
  s
end

function Snapshots(values::AbstractParamContainer,args...)
  BasicSnapshots(values,args...)
end

num_space_dofs(s::BasicSnapshots) = length(first(s.values))

function change_mode(s::BasicSnapshots{Mode1Axis})
  BasicSnapshots(s.values,s.realization,Mode2Axis())
end

function change_mode(s::BasicSnapshots{Mode2Axis})
  BasicSnapshots(s.values,s.realization,Mode1Axis())
end

function tensor_getindex(s::BasicSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace]
end

function tensor_setindex!(s::BasicSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace] = v
end

struct TransientSnapshots{M,T,P,R,V} <: StandardSnapshots{M,T}
  mode::M
  values::V
  realization::R
  function TransientSnapshots(
    values::AbstractVector{P},
    realization::R,
    mode::M=Mode1Axis(),
    ) where {M,P<:AbstractParamContainer,R<:TransientParamRealization}

    V = typeof(values)
    T = eltype(P)
    new{M,T,P,R,V}(mode,values,realization)
  end
end

function Snapshots(values::AbstractVector{P},args...) where P<:AbstractParamContainer
  TransientSnapshots(values,args...)
end

num_space_dofs(s::TransientSnapshots) = length(first(first(s.values)))

function change_mode(s::TransientSnapshots{Mode1Axis})
  TransientSnapshots(s.values,s.realization,Mode2Axis())
end

function change_mode(s::TransientSnapshots{Mode2Axis})
  TransientSnapshots(s.values,s.realization,Mode1Axis())
end

function tensor_getindex(s::TransientSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace]
end

function tensor_setindex!(s::TransientSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace] = v
end

function BasicSnapshots(
  s::TransientSnapshots{M,T,<:ParamArray{T,N,L,A}}
  ) where {M,T,N,L,A}

  nt = num_times(s)
  np = num_params(s)
  array = Vector{eltype(A)}(undef,nt*np)
  @inbounds for i = 1:nt*np
    it = slow_index(i,np)
    ip = fast_index(i,np)
    array[i] = s.values[it][ip]
  end
  basic_values = ParamArray(array)
  BasicSnapshots(basic_values,s.realization,s.mode)
end

function FEM.get_values(s::TransientSnapshots)
  get_values(BasicSnapshots(s))
end

struct CompressedTransientSnapshots{M,N,T,R,V} <: StandardSnapshots{M,T}
  current_mode::M
  initial_mode::N
  values::V
  realization::R
  function CompressedTransientSnapshots(
    values::AbstractMatrix{T},
    realization::R,
    current_mode::M=Mode1Axis(),
    initial_mode::N=Mode1Axis()) where {M,N,T,R}

    V = typeof(values)
    new{M,N,T,R,V}(current_mode,initial_mode,values,realization)
  end
end

function Snapshots(values::AbstractMatrix{T},args...) where T<:Number
  CompressedTransientSnapshots(values,args...)
end

num_space_dofs(s::CompressedTransientSnapshots{M,Mode1Axis}) where M = size(s.values,1)
num_space_dofs(s::CompressedTransientSnapshots{M,Mode2Axis}) where M = Int(size(s.values,2) / num_params(s))

function change_mode(s::CompressedTransientSnapshots{Mode1Axis})
  CompressedTransientSnapshots(s.values,s.realization,Mode2Axis(),Mode1Axis())
end

function change_mode(s::CompressedTransientSnapshots{Mode2Axis})
  CompressedTransientSnapshots(s.values,s.realization,Mode1Axis(),Mode2Axis())
end

column_index(a,b,Na,Nb) = (a-1)*Nb+b
column_index(a::Colon,b,Na,Nb) = b:Nb:Na*Nb
column_index(a,b::Colon,Na,Nb) = a:Na:Na*Nb
column_index(a::Colon,b::Colon,Na,Nb) = a

function tensor_getindex(
  s::CompressedTransientSnapshots{M,Mode1Axis},
  ispace,itime,iparam
  ) where M

  nt = num_times(s)
  np = num_params(s)
  icolumn = column_index(itime,iparam,nt,np)
  s.values[ispace,icolumn]
end
function tensor_getindex(
  s::CompressedTransientSnapshots{M,Mode2Axis},
  ispace,itime,iparam
  ) where M

  ns = num_space_dofs(s)
  np = num_params(s)
  icolumn = column_index(ispace,iparam,ns,np)
  s.values[itime,icolumn]
end

function tensor_setindex!(
  s::CompressedTransientSnapshots{M,Mode1Axis},
  v,ispace,itime,iparam
  ) where M

  nt = num_times(s)
  np = num_params(s)
  icolumn = column_index(itime,iparam,nt,np)
  s.values[ispace,icolumn] = v
end
function tensor_setindex!(
  s::CompressedTransientSnapshots{M,Mode2Axis},
  v,ispace,itime,iparam
  ) where M

  ns = num_space_dofs(s)
  np = num_params(s)
  icolumn = column_index(ispace,iparam,ns,np)
  s.values[itime,icolumn] = v
end

struct SelectedSnapshotsAtIndices{M,T,S,I} <: StandardSnapshots{M,T}
  snaps::S
  selected_indices::I
  function SelectedSnapshotsAtIndices(
    snaps::StandardSnapshots{M,T},
    selected_indices::I
    ) where {M,T,I}

    S = typeof(snaps)
    new{M,T,S,I}(snaps,selected_indices)
  end
end

function SelectedSnapshotsAtIndices(s::SelectedSnapshotsAtIndices,selected_indices)
  new_srange,new_trange,new_prange = selected_indices
  old_srange,old_trange,old_prange = s.selected_indices
  @check intersect(old_srange,new_srange) == new_srange
  @check intersect(old_trange,new_trange) == new_trange
  @check intersect(old_prange,new_prange) == new_prange
  SelectedSnapshotsAtIndices(s.snaps,selected_indices)
end

function select_snapshots(s::AbstractSnapshots,spacerange,timerange,paramrange)
  srange = isa(spacerange,Colon) ? Base.OneTo(num_space_dofs(s)) : spacerange
  srange = isa(srange,Integer) ? [srange] : srange
  trange = isa(timerange,Colon) ? Base.OneTo(num_times(s)) : timerange
  trange = isa(trange,Integer) ? [trange] : trange
  prange = isa(paramrange,Colon) ? Base.OneTo(num_params(s)) : paramrange
  prange = isa(prange,Integer) ? [prange] : prange
  selected_indices = (srange,trange,prange)
  SelectedSnapshotsAtIndices(s,selected_indices)
end

function select_snapshots(s::AbstractSnapshots,timerange,paramrange;spacerange=:)
  select_snapshots(s,spacerange,timerange,paramrange)
end

function select_snapshots(s::AbstractSnapshots,paramrange;spacerange=:,timerange=:)
  select_snapshots(s,spacerange,timerange,paramrange)
end

function select_snapshots(s::AbstractSnapshots;kwargs...)
  paramrange = isa(s,SelectedSnapshotsAtIndices) ? last(s.selected_indices) : Colon()
  select_snapshots(s,paramrange;kwargs...)
end

function select_snapshots_entries(s::StandardSnapshots{M,T},spacerange,timerange) where {M,T}
  ss = select_snapshots(s,spacerange,timerange,Base.OneTo(num_params(s))) |> copy
  nval = length(spacerange),length(timerange)
  nt = length(timerange)
  np = num_params(s)
  values = allocate_param_array(zeros(T,nval),np)
  @inbounds for ip = 1:np
    cols = (ip-1)*nt+1:ip*nt
    values[ip] = ss[:,cols]
  end
  return values
end

space_indices(s::SelectedSnapshotsAtIndices) = s.selected_indices[1]
time_indices(s::SelectedSnapshotsAtIndices) = s.selected_indices[2]
param_indices(s::SelectedSnapshotsAtIndices) = s.selected_indices[3]
num_space_dofs(s::SelectedSnapshotsAtIndices) = length(space_indices(s))
FEM.num_times(s::SelectedSnapshotsAtIndices) = length(time_indices(s))
FEM.num_params(s::SelectedSnapshotsAtIndices) = length(param_indices(s))

function change_mode(s::SelectedSnapshotsAtIndices)
  snaps = change_mode(s.snaps)
  SelectedSnapshotsAtIndices(snaps,s.selected_indices)
end

function tensor_getindex(
  s::SelectedSnapshotsAtIndices,ispace,itime,iparam)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_getindex(s.snaps,is,it,ip)
end

function tensor_setindex!(
  s::SelectedSnapshotsAtIndices,v,ispace,itime,iparam)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedSnapshotsAtIndices{Mode1Axis,T,<:BasicSnapshots}) where T
  v = get_values(s.snaps)
  array = Vector{typeof(first(v))}(undef,num_cols(s))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      array[(i-1)*num_params(s)+j] = v[(it-1)*num_params(s)+jp][space_indices(s)]
    end
  end
  ParamArray(array)
end

function FEM.get_values(s::SelectedSnapshotsAtIndices{M,T,<:TransientSnapshots}) where {M,T}
  get_values(BasicSnapshots(s))
end

function get_realization(s::SelectedSnapshotsAtIndices)
  r = get_realization(s.snaps)
  r[param_indices(s),time_indices(s)]
end

get_mode(s::SelectedSnapshotsAtIndices) = get_mode(s.snaps)

function BasicSnapshots(s::SelectedSnapshotsAtIndices{M,T,<:BasicSnapshots}) where {M,T}
  values = get_values(s)
  r = get_realization(s)
  mode = s.snaps.mode
  BasicSnapshots(values,r,mode)
end

function BasicSnapshots(s::SelectedSnapshotsAtIndices{M,T,<:TransientSnapshots}) where {M,T}
  v = s.snaps.values
  basic_values = Vector{typeof(first(first(v)))}(undef,num_cols(s))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      basic_values[(i-1)*num_params(s)+j] = v[it][jp][space_indices(s)]
    end
  end
  r = get_realization(s)
  mode = s.snaps.mode
  BasicSnapshots(ParamArray(basic_values),r,mode)
end

#= mode-1 representation of a snapshot of type SnapshotsSwappedColumns
   [ [u(x1,t1,μ1) ⋯ u(x1,tT,μ1)] [u(x1,t1,μ2) ⋯ u(x1,tT,μ2)] [u(x1,t1,μ3) ⋯] [⋯] [u(x1,t1,μP) ⋯ u(x1,tT,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(xN,t1,μ1) ⋯ u(xN,tT,μ1)] [u(xN,t1,μ2) ⋯ u(xN,tT,μ2)] [u(xN,t1,μ3) ⋯] [⋯] [u(xN,t1,μP) ⋯ u(xN,tT,μP)] ]
=#
#= mode-2 representation of a snapshot of type SnapshotsSwappedColumns
   [ [u(x1,t1,μ1) ⋯ u(xN,t1,μ1)] [u(x1,t1,μ2) ⋯ u(xN,t1,μ2)] [u(x1,t1,μ3) ⋯] [⋯] [u(x1,t1,μP) ⋯ u(xN,t1,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(x1,tT,μ1) ⋯ u(xN,tT,μ1)] [u(x1,tT,μ2) ⋯ u(xN,tT,μ2)] [u(x1,tT,μ3) ⋯] [⋯] [u(x1,tT,μP) ⋯ u(xN,tT,μP)] ]
=#

abstract type SnapshotsSwappedColumns{T} <: StandardSnapshots{Mode1Axis,T} end

num_space_dofs(s::SnapshotsSwappedColumns) = num_space_dofs(s.snaps)

get_realization(s::SnapshotsSwappedColumns) = get_realization(s.snaps)

function col_index(s::SnapshotsSwappedColumns,time_index::Integer,param_index::Integer)
  (param_index-1)*num_times(s)+time_index
end

function change_mode(s::SnapshotsSwappedColumns)
  @notimplemented
end

function Base.getindex(s::SnapshotsSwappedColumns,ispace,j)
  nt = num_times(s)
  iparam = slow_index(j,nt)
  itime = fast_index(j,nt)
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.setindex!(s::SnapshotsSwappedColumns,v,ispace,j)
  nt = num_times(s)
  iparam = slow_index(j,nt)
  itime = fast_index(j,nt)
  tensor_setindex!(s,v,ispace,itime,iparam)
end

function tensor_getindex(s::SnapshotsSwappedColumns,ispace,itime,iparam)
  tensor_getindex(s.snaps,ispace,itime,iparam)
end

function tensor_setindex!(s::SnapshotsSwappedColumns,v,ispace,itime,iparam)
  tensor_setindex!(s.snaps,v,ispace,itime,iparam)
end

function reverse_snapshots(s::AbstractSnapshots)
  @abstractmethod
end

function reverse_snapshots(s::SnapshotsSwappedColumns)
  s
end

function reverse_snapshots(s::BasicSnapshots)
  BasicSnapshotsSwappedColumns(s)
end

function reverse_snapshots(s::TransientSnapshots)
  TransientSnapshotsSwappedColumns(s)
end

function reverse_snapshots(s::SelectedSnapshotsAtIndices)
  SelectedSnapshotsSwappedColumns(s)
end

function reverse_snapshots(values,r::TransientParamRealization)
  snaps = Snapshots(values,r)
  reverse_snapshots(snaps)
end

function reverse_snapshots(a::ArrayContribution,args...)
  contribution(a.trians) do trian
    reverse_snapshots(a[trian],args...)
  end
end

function SelectedSnapshotsAtIndices(
  s::SnapshotsSwappedColumns,selected_indices::Tuple)
  SelectedSnapshotsSwappedColumns(s,selected_indices)
end

struct BasicSnapshotsSwappedColumns{T,S} <: SnapshotsSwappedColumns{T}
  snaps::S
  function BasicSnapshotsSwappedColumns(snaps::S) where S<:BasicSnapshots
    T = eltype(snaps)
    new{T,S}(snaps)
  end
end

function FEM.get_values(s::BasicSnapshotsSwappedColumns)
  nt = num_times(s)
  np = num_params(s)
  T = typeof(first(s.snaps.values))
  array = Vector{T}(undef,nt*np)
  @inbounds for i = 1:nt*np
    it = fast_index(i,nt)
    ip = slow_index(i,nt)
    array[i] = s.snaps.values[(it-1)*np+ip]
  end
  return ParamArray(copy(array))
end

struct TransientSnapshotsSwappedColumns{T,S} <: SnapshotsSwappedColumns{T}
  snaps::S
  function TransientSnapshotsSwappedColumns(snaps::S) where S<:TransientSnapshots
    T = eltype(snaps)
    new{T,S}(snaps)
  end
end

function FEM.get_values(s::TransientSnapshotsSwappedColumns)
  nt = num_times(s)
  np = num_params(s)
  T = typeof(first(first(s.snaps.values)))
  array = Vector{T}(undef,nt*np)
  @inbounds for i = 1:nt*np
    it = fast_index(i,nt)
    ip = slow_index(i,nt)
    array[i] = s.values[it][ip]
  end
  return ParamArray(copy(array))
end

struct SelectedSnapshotsSwappedColumns{T,S,I} <: SnapshotsSwappedColumns{T}
  snaps::S
  selected_indices::I
  function SelectedSnapshotsSwappedColumns(
    s::SnapshotsSwappedColumns{T},
    selected_indices::I
    ) where {T,I}

    S = typeof(s.snaps)
    new{T,S,I}(s.snaps,selected_indices)
  end
end

function SelectedSnapshotsSwappedColumns(s::SelectedSnapshotsAtIndices)
  sswap = reverse_snapshots(s.snaps)
  SelectedSnapshotsSwappedColumns(sswap,s.selected_indices)
end

space_indices(s::SelectedSnapshotsSwappedColumns) = s.selected_indices[1]
time_indices(s::SelectedSnapshotsSwappedColumns) = s.selected_indices[2]
param_indices(s::SelectedSnapshotsSwappedColumns) = s.selected_indices[3]
num_space_dofs(s::SelectedSnapshotsSwappedColumns) = length(space_indices(s))
FEM.num_times(s::SelectedSnapshotsSwappedColumns) = length(time_indices(s))
FEM.num_params(s::SelectedSnapshotsSwappedColumns) = length(param_indices(s))

function tensor_getindex(
  s::SelectedSnapshotsSwappedColumns,ispace,itime,iparam)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_getindex(s.snaps,is,it,ip)
end

function tensor_setindex!(
  s::SelectedSnapshotsSwappedColumns,v,ispace,itime,iparam)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedSnapshotsSwappedColumns)
  v = get_values(s.snaps)
  values = Vector{typeof(first(v))}(undef,num_cols(s))
  for (i,ip) in enumerate(param_indices(s))
    for (j,jt) in enumerate(time_indices(s))
      @inbounds values[(i-1)*num_times(s)+j] = v[(ip-1)*num_times(s)+jt][space_indices(s)]
    end
  end
  ParamArray(copy(values))
end

function get_realization(s::SelectedSnapshotsSwappedColumns)
  r = get_realization(s.snaps)
  r[param_indices(s),time_indices(s)]
end

const BasicNnzSnapshots = BasicSnapshots{M,T,P,R} where {M,T,P<:ParamSparseMatrix,R}
const TransientNnzSnapshots = TransientSnapshots{M,T,P,R} where {M,T,P<:ParamSparseMatrix,R}
const SelectedNnzSnapshotsAtIndices = SelectedSnapshotsAtIndices{M,T,S,I} where {M,T,S<:Union{BasicNnzSnapshots,TransientNnzSnapshots},I}
const GenericNnzSnapshots = Union{
  BasicNnzSnapshots{M,T},
  TransientNnzSnapshots{M,T},
  SelectedNnzSnapshotsAtIndices{M,T}} where {M,T}

const BasicNnzSnapshotsSwappedColumns = BasicSnapshotsSwappedColumns{T,S} where {T,S<:BasicNnzSnapshots}
const TransientNnzSnapshotsSwappedColumns = TransientSnapshotsSwappedColumns{T,S} where {T,S<:TransientNnzSnapshots}
const SelectedNnzSnapshotsSwappedColumns = SelectedSnapshotsSwappedColumns{T,S} where {T,S<:Union{BasicNnzSnapshots,TransientNnzSnapshots}}
const NnzSnapshotsSwappedColumns = Union{
  BasicNnzSnapshotsSwappedColumns{T},
  TransientNnzSnapshotsSwappedColumns{T},
  SelectedNnzSnapshotsSwappedColumns{T}
} where T

const NnzSnapshots = Union{
  GenericNnzSnapshots{M,T},
  NnzSnapshotsSwappedColumns{T}} where {M,T}

num_space_dofs(s::BasicNnzSnapshots) = nnz(first(s.values))
num_full_space_dofs(s::BasicNnzSnapshots) = length(first(s.values))

function tensor_getindex(s::BasicNnzSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[ispace]
end

num_space_dofs(s::TransientNnzSnapshots) = nnz(first(first(s.values)))
num_full_space_dofs(s::TransientNnzSnapshots) = length(first(first(s.values)))

function tensor_getindex(
  s::TransientNnzSnapshots,
  ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[itime][iparam])[ispace]
end

function tensor_getindex(
  s::BasicNnzSnapshotsSwappedColumns,
  ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.snaps.values[(itime-1)*num_params(s)+iparam])[ispace]
end

function tensor_getindex(
  s::TransientNnzSnapshotsSwappedColumns,
  ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.snaps.values[itime][iparam])[ispace]
end

function tensor_getindex(
  s::SelectedNnzSnapshotsSwappedColumns{T,<:BasicNnzSnapshots},
  ispace::Integer,itime::Integer,iparam::Integer) where T
  snaps = s.snaps
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  nonzeros(snaps.values[(it-1)*num_params(s)+ip])[is]
end

function tensor_getindex(
  s::SelectedNnzSnapshotsSwappedColumns{T,<:TransientNnzSnapshots},
  ispace::Integer,itime::Integer,iparam::Integer) where T
  snaps = s.snaps
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  nonzeros(snaps.values[ip][it])[is]
end

sparsify_indices(s::BasicNnzSnapshots,srange::AbstractVector) = sparsify_indices(first(s.values),srange)
sparsify_indices(s::TransientNnzSnapshots,srange::AbstractVector) = sparsify_indices(first(first(s.values)),srange)
sparsify_indices(s::NnzSnapshotsSwappedColumns,srange::AbstractVector) = sparsify_indices(s.snaps,srange)

function select_snapshots(s::NnzSnapshots,spacerange,timerange,paramrange)
  _srange = isa(spacerange,Colon) ? Base.OneTo(num_space_dofs(s)) : spacerange
  _srange = isa(_srange,Integer) ? [_srange] : _srange
  srange = sparsify_indices(s,_srange)
  trange = isa(timerange,Colon) ? Base.OneTo(num_times(s)) : timerange
  trange = isa(trange,Integer) ? [trange] : trange
  prange = isa(paramrange,Colon) ? Base.OneTo(num_params(s)) : paramrange
  prange = isa(prange,Integer) ? [prange] : prange
  selected_indices = (srange,trange,prange)
  SelectedSnapshotsAtIndices(s,selected_indices)
end

function recast(s::NnzSnapshots{Mode1Axis},a::AbstractMatrix)
  v = isa(s,BasicSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  m,n = size(v)
  asparse = map(eachcol(a)) do v
    sparse(i,j,v,m,n)
  end
  return VecOfSparseMat2Mat(asparse)
end

struct VecOfSparseMat2Mat{Tv,Ti,V} <: AbstractMatrix{Tv}
  values::V
  function VecOfSparseMat2Mat(values::V) where {Tv,Ti,V<:AbstractVector{<:AbstractSparseMatrix{Tv,Ti}}}
    new{Tv,Ti,V}(values)
  end
end

FEM.get_values(s::VecOfSparseMat2Mat) = s.values
Base.size(s::VecOfSparseMat2Mat) = (nnz(first(s.values)),length(s.values))

function Base.getindex(s::VecOfSparseMat2Mat,i,j::Integer)
  nonzeros(s.values[j])[i]
end

function Base.getindex(s::VecOfSparseMat2Mat,i,j)
  view(s,i,j)
end

function get_nonzero_indices(s::VecOfSparseMat2Mat)
  get_nonzero_indices(first(s.values))
end

struct BlockSnapshots{S,N} <: AbstractParamContainer{S,N}
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

function BlockSnapshots(k::BlockMap{N},a::S...) where {S<:AbstractSnapshots,N}
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

function Snapshots(values::ParamBlockArray,args...)
  block_values = blocks(values)
  nblocks = blocksize(values)
  active_block_ids = findall(!iszero,block_values)
  block_map = BlockMap(nblocks,active_block_ids)
  active_block_snaps = Any[Snapshots(block_values[i],args...) for i in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps...)
end

function Snapshots(values::AbstractVector{<:ParamBlockArray},args...)
  vals = first(values)
  block_vals = blocks(vals)
  nblocks = blocksize(vals)
  active_block_ids = findall(!iszero,block_vals)
  block_map = BlockMap(nblocks,active_block_ids)
  vec_block_values = map(blocks,values)
  active_block_snaps = Any[Snapshots(map(x->x[i],vec_block_values),args...) for i in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps...)
end

Base.size(s::BlockSnapshots,i...) = size(s.array,i...)
Base.length(s::BlockSnapshots) = length(s.array)
Base.eltype(::Type{<:BlockSnapshots{S}}) where S = S
Base.eltype(s::BlockSnapshots{S}) where S = S
Base.ndims(s::BlockSnapshots{S,N}) where {S,N} = N
Base.ndims(::Type{BlockSnapshots{S,N}}) where {S,N} = N
Base.copy(s::BlockSnapshots) = BlockSnapshots(copy(s.array),copy(s.touched))
Base.eachindex(s::BlockSnapshots) = eachindex(s.array)
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

FEM.num_times(s::BlockSnapshots) = num_times(testitem(s))
FEM.num_params(s::BlockSnapshots) = num_params(testitem(s))

function FEM.get_values(s::BlockSnapshots)
  map(get_values,s.array) |> mortar
end

function get_realization(s::BlockSnapshots)
  get_realization(testitem(s))
end

function get_mode(s::BlockSnapshots)
  get_mode(testitem(s))
end

function get_touched_blocks(s::ArrayBlock)
  findall(s.touched)
end

function get_touched_blocks(s::BlockSnapshots)
  findall(s.touched)
end

function get_touched_linear_blocks(s::BlockSnapshots)
  ids = get_touched_blocks(s)
  isa(first(ids),CartesianIndex) ? Tuple.(ids) : ids
end

function change_mode(s::BlockSnapshots{<:Any,N},args...;kwargs...) where N
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  active_block_snaps = Any[change_mode(s[i],args...;kwargs...) for i in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps...)
end

function select_snapshots(s::BlockSnapshots{<:Any,N},args...;kwargs...) where N
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  active_block_snaps = Any[select_snapshots(s[i],args...;kwargs...) for i in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps...)
end

function select_snapshots_entries(
  s::BlockSnapshots{<:Any,N},
  spacerange::ArrayBlock{<:Any,N},
  timerange) where N

  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  active_block_snaps = Any[select_snapshots_entries(s[i],spacerange[i],timerange) for i in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps...)
end

function reverse_snapshots(s::BlockSnapshots{<:Any,N}) where N
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  active_block_snaps = Any[reverse_snapshots(s[i]) for i in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps...)
end

function reverse_snapshots(
  values::AbstractVector{P},
  r::TransientParamRealization
  ) where P<:ParamBlockArray

  vals = first(values)
  block_vals = blocks(vals)
  nblocks = blocksize(vals)
  active_block_ids = findall(!iszero,block_vals)
  block_map = BlockMap(nblocks,active_block_ids)
  vec_block_values = map(blocks,values)
  active_block_snaps = Any[reverse_snapshots(map(x->x[i],vec_block_values),r) for i in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps...)
end

struct TensorTrainSnapshots{S,R}
  snaps::S
  ranks::R
end

const AbstractSubTransientSnapshots = Union{A,SubArray{T,N,A}} where {T,N,A<:AbstractSnapshots{T,N}}

function (*)(
  a::Adjoint{<:Any,<:AbstractSubTransientSnapshots},
  b::AbstractMatrix
  )

  (*)(collect(a),b)
end

function (*)(
  a::Adjoint{<:Any,<:AbstractVecOrMat},
  b::AbstractSubTransientSnapshots
  )

  (*)(a,collect(b))
end

function (*)(
  a::Adjoint{<:Any,<:AbstractSubTransientSnapshots},
  b::AbstractSubTransientSnapshots
  )

  (*)(collect(a),collect(b))
end

function (*)(
  a::Adjoint{<:Any,<:AbstractSubTransientSnapshots},
  b::Diagonal
  )

  (*)(collect(a),b)
end

function (*)(
  a::Diagonal,
  b::AbstractSubTransientSnapshots
  )

  (*)(a,collect(b))
end

for op in (:*,:\)
  @eval begin
    function ($op)(
      a::A,
      b::AbstractMatrix
      ) where A<:AbstractSubTransientSnapshots

      ($op)(a,collect(b))
    end

    function ($op)(
      a::AbstractMatrix,
      b::A
      ) where A<:AbstractSubTransientSnapshots

      ($op)(a,collect(b))
    end

    function ($op)(
      a::A,
      b::B
      ) where {A<:AbstractSubTransientSnapshots,B<:AbstractSubTransientSnapshots}

      ($op)(collect(a),collect(b))
    end
  end
end
