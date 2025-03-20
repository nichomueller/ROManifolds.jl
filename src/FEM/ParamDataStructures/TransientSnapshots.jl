"""
    abstract type TransientSnapshots{T,N,I,R<:TransientRealization,A} <: Snapshots{T,N,I,R,A} end

Transient specialization of a `Snapshots`

Subtypes:
- [`TransientGenericSnapshots`](@ref)
- [`GenericSnapshots`](@ref)
- [`TransientSnapshotsAtIndices`](@ref)
- [`TransientSnapshotsWithIC`](@ref)
- [`ModeTransientSnapshots`](@ref)
"""
abstract type TransientSnapshots{T,N,I,R<:TransientRealization,A} <: Snapshots{T,N,I,R,A} end

space_dofs(s::TransientSnapshots{T,N}) where {T,N} = size(get_all_data(s))[1:N-2]

num_times(s::TransientSnapshots) = num_times(get_realization(s))

Base.size(s::TransientSnapshots) = (space_dofs(s)...,num_times(s),num_params(s))

get_initial_data(s::TransientSnapshots) = @abstractmethod

function get_mode1(s::TransientSnapshots)
  data = get_all_data(s)
  reshape(data,num_space_dofs(s),:)
end

function get_mode2(s::TransientSnapshots)
  mode1 = get_mode1(s)
  change_mode(mode1,num_params(s))
end

function select_snapshots(s::TransientSnapshots,prange,trange=1:num_times(s))
  drange = view(get_all_data(s),:,prange,trange)
  rrange = get_realization(s)[prange,trange]
  Snapshots(drange,get_dof_map(s),rrange)
end

"""
    struct TransientSnapshotsWithIC{T,N,I,R,A,B<:TransientSnapshots{T,N,I,R,A}} <: TransientSnapshots{T,N,I,R,A}
      initial_data::A
      snaps::B
    end

Stores a [`TransientSnapshots`](@ref) `snaps` alongside a parametric initial condition `initial_data`
"""
struct TransientSnapshotsWithIC{T,N,I,R,A,B<:TransientSnapshots{T,N,I,R}} <: TransientSnapshots{T,N,I,R,A}
  initial_data::A
  snaps::B
end

function Snapshots(s,s0::AbstractParamArray,i::AbstractDofMap,r::TransientRealization)
  snaps = Snapshots(s,i,r)
  TransientSnapshotsWithIC(s0,snaps)
end

get_all_data(s::TransientSnapshotsWithIC) = get_all_data(s.snaps)
get_initial_data(s::TransientSnapshotsWithIC) = s.initial_data
get_param_data(s::TransientSnapshotsWithIC) = get_param_data(s.snaps)
DofMaps.get_dof_map(s::TransientSnapshotsWithIC) = get_dof_map(s.snaps)
get_realization(s::TransientSnapshotsWithIC) = get_realization(s.snaps)

function Base.getindex(s::TransientSnapshotsWithIC{T,N},i::Vararg{Integer,N}) where {T,N}
  getindex(s.snaps,i...)
end

function Base.setindex!(s::TransientSnapshotsWithIC{T,N},v,i::Vararg{Integer,N}) where {T,N}
  setindex!(s.snaps,v,i...)
end

# sparse interface

function Snapshots(s::ParamSparseMatrix,i::SparseMatrixDofMap,r::TransientRealization)
  T = eltype(s)
  i = get_dof_map(s)
  data = get_all_data(s)
  idata = zeros(T,size(i)...,num_params(r),num_times(r))
  for it in 1:num_times(r), ip in 1:num_params(r)
    ipt = (it-1)*num_params(r)+ip
    for k in CartesianIndices(i)
      k′ = i[k]
      if k′ > 0
        idata[k.I...,ip,it] = data[k′,ipt]
      end
    end
  end
  Snapshots(idata,i,r)
end

# block snapshots

function Snapshots(
  data::BlockParamArray{T,N},
  data0::BlockParamArray,
  i::AbstractArray{<:AbstractDofMap},
  r::AbstractRealization
  ) where {T,N}

  block_values = blocks(data)
  block_value0 = blocks(data0)
  s = size(block_values)
  @check s == size(i)

  array = Array{Snapshots,N}(undef,s)
  touched = Array{Bool,N}(undef,s)
  for j in 1:length(block_values)
    dataj = block_values[j]
    data0j = block_value0[j]
    if !iszero(dataj)
      array[j] = Snapshots(dataj,data0j,i[j],r)
      touched[j] = true
    else
      touched[j] = false
    end
  end

  BlockSnapshots(array,touched)
end

function Arrays.return_cache(::typeof(get_initial_data),s::BlockSnapshots{S,N}) where {S,N}
  cache = get_initial_data(testitem(s))
  block_cache = Array{typeof(cache),N}(undef,size(s))
  return block_cache
end

function get_initial_data(s::BlockSnapshots)
  values = return_cache(get_initial_data,s)
  for i in eachindex(s.touched)
    if s.touched[i]
      values[i] = get_initial_data(s[i])
    end
  end
  return mortar(values)
end

# utils

function Snapshots(
  a::TupOfArrayContribution,
  i::TupOfArrayContribution,
  r::AbstractRealization)

  map((a,i)->Snapshots(a,i,r),a,i)
end

function select_snapshots(a::TupOfArrayContribution,args...;kwargs...)
  map(a->select_snapshots(a,args...;kwargs...),a)
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
