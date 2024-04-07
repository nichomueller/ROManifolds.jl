function _minimum_dir_d(i::AbstractVector{CartesianIndex{D}},d::Integer) where D
  mind = Inf
  for ii in i
    if ii.I[d] < mind
      mind = ii.I[d]
    end
  end
  return mind
end

function _maximum_dir_d(i::AbstractVector{CartesianIndex{D}},d::Integer) where D
  maxd = 0
  for ii in i
    if ii.I[d] > maxd
      maxd = ii.I[d]
    end
  end
  return maxd
end

function _shape_per_dir(i::AbstractVector{CartesianIndex{D}}) where D
  function _admissible_shape(d::Int)
    mind = _minimum_dir_d(i,d)
    maxd = _maximum_dir_d(i,d)
    @assert all([ii.I[d] ≥ mind for ii in i]) && all([ii.I[d] ≤ maxd for ii in i])
    return maxd - mind + 1
  end
  ntuple(d -> _admissible_shape(d),D)
end

function get_dof_permutation(
  model::CartesianDiscreteModel{Dc},
  space::UnconstrainedFESpace,
  order::Integer) where Dc

  function get_terms(p::Polytope,orders)
    _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
    terms = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
    return terms
  end

  desc = get_cartesian_descriptor(model)

  periodic = desc.isperiodic
  ncells = desc.partition
  ndofs = order .* ncells .+ 1 .- periodic

  new_dof_ids = copy(LinearIndices(ndofs))

  terms = get_terms(first(get_polytopes(model)),fill(order,Dc))
  cell_dof_ids = get_cell_dof_ids(space)
  cache_cell_dof_ids = array_cache(cell_dof_ids)

  for (icell,cell) in enumerate(CartesianIndices(ncells))
    first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
    new_dofs_range = map(i -> i:i+order,first_new_dof)
    new_dofs = view(new_dof_ids,new_dofs_range...)

    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    for (idof,dof) in enumerate(cell_dofs)
      t = terms[idof]
      new_dofs[t] < 0 && continue
      if dof < 0
        new_dofs[t] *= -1
      end
    end
  end

  pos_ids = findall(new_dof_ids.>0)
  neg_ids = findall(new_dof_ids.<0)
  new_dof_ids[pos_ids] .= LinearIndices(pos_ids)
  new_dof_ids[neg_ids] .= -1 .* LinearIndices(neg_ids)

  free_vals_shape = _shape_per_dir(pos_ids)
  n2o_dof_map = fill(-1,free_vals_shape)

  for (icell,cell) in enumerate(CartesianIndices(ncells))
    first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
    new_dofs_range = map(i -> i:i+order,first_new_dof)
    new_dofs = view(new_dof_ids,new_dofs_range...)

    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    for (idof,dof) in enumerate(cell_dofs)
      t = terms[idof]
      new_dofs[t] < 0 && continue
      n2o_dof_map[new_dofs[t]] = dof
    end
  end

  return n2o_dof_map
end

function get_dof_permutation(space::FESpace)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  order = FEM.get_polynomial_order(space)
  get_dof_permutation(model,space,order)
end

function get_dof_permutation(feop::TransientFEOperator)
  get_dof_permutation(get_test(feop))
end

abstract type TTSnapshots{T,N} <: AbstractSnapshots{T,N} end

Base.size(s::TTSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))
Base.length(s::TTSnapshots) = prod(size(s))
Base.axes(s::TTSnapshots) = Base.OneTo.(size(s))

get_permutation(s::TTSnapshots) = s.permutation

num_space_dofs(s::TTSnapshots) = size(get_permutation(s))

function Base.getindex(s::TTSnapshots{T,3},ix,itime,iparam) where T
  view(s,ix,itime,iparam)
end
function Base.getindex(s::TTSnapshots{T,4},ix,iy,itime,iparam) where T
  view(s,ix,iy,itime,iparam)
end
function Base.getindex(s::TTSnapshots{T,5},ix,iy,iz,itime,iparam) where T
  view(s,ix,iy,iz,itime,iparam)
end

function Base.getindex(s::TTSnapshots{T,3},ix::Integer,itime::Integer,iparam::Integer) where T
  tensor_getindex(s,CartesianIndex(ix),itime,iparam)
end
function Base.getindex(s::TTSnapshots{T,4},ix::Integer,iy::Integer,itime::Integer,iparam::Integer) where T
  tensor_getindex(s,CartesianIndex(ix,iy),itime,iparam)
end
function Base.getindex(s::TTSnapshots{T,5},ix::Integer,iy::Integer,iz::Integer,itime::Integer,iparam::Integer) where T
  tensor_getindex(s,CartesianIndex(ix,iy,iz),itime,iparam)
end

function Base.setindex!(s::TTSnapshots{T,3},v,ix::Integer,itime::Integer,iparam::Integer) where T
  tensor_setindex!(s,v,CartesianIndex(ix),itime,iparam)
end
function Base.setindex!(s::TTSnapshots{T,4},v,ix::Integer,iy::Integer,itime::Integer,iparam::Integer) where T
  tensor_setindex!(s,v,CartesianIndex(ix,iy),itime,iparam)
end
function Base.setindex!(s::TTSnapshots{T,5},v,ix::Integer,iy::Integer,iz::Integer,itime::Integer,iparam::Integer) where T
  tensor_setindex!(s,v,CartesianIndex(ix,iy,iz),itime,iparam)
end

reverse_snapshots(s::TTSnapshots) = s

#= representation of a standard tensor-train snapshot
   [ [u(x1,t1,μ1) ⋯ u(x1,t1,μP)] [u(x1,t2,μ1) ⋯ u(x1,t2,μP)] [u(x1,t3,μ1) ⋯] [⋯] [u(x1,tT,μ1) ⋯ u(x1,tT,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(xN,t1,μ1) ⋯ u(xN,t1,μP)] [u(xN,t2,μ1) ⋯ u(xN,t2,μP)] [u(xN,t3,μ1) ⋯] [⋯] [u(xN,tT,μ1) ⋯ u(xN,tT,μP)] ]
=#

struct BasicTTSnapshots{T,N,P,R,D} <: TTSnapshots{T,N}
  values::P
  realization::R
  permutation::Array{Int,D}
  function BasicTTSnapshots(values::P,realization::R,perm::Array{Int,D}) where {P<:ParamArray,R,D}
    T = eltype(P)
    N = D+2
    new{T,N,P,R,D}(values,realization,perm)
  end
end

function BasicSnapshots(
  values::ParamArray,
  realization::TransientParamRealization,
  perm::Array{Int})
  BasicTTSnapshots(values,realization,perm)
end

function BasicSnapshots(s::BasicTTSnapshots)
  s
end

# num_space_dofs_1D(s::BasicTTSnapshots{T,N}) where {T,N} = length(first(s.values))

function tensor_getindex(s::BasicTTSnapshots,ispace,itime,iparam)
  perm_ispace = s.permutation[ispace]
  s.values[iparam+(itime-1)*num_params(s)][perm_ispace]
end

function tensor_setindex!(s::BasicTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = s.permutation[ispace]
  s.values[iparam+(itime-1)*num_params(s)][perm_ispace] = v
end

struct TransientTTSnapshots{T,N,P,R,V,D} <: TTSnapshots{T,N}
  values::V
  realization::R
  permutation::Array{Int,D}
  function TransientTTSnapshots(
    values::AbstractVector{P},
    realization::R,
    perm::Array{Int,D}
    ) where {P<:ParamArray,R<:TransientParamRealization,D}

    V = typeof(values)
    T = eltype(P)
    N = D+2
    new{T,N,P,R,V,D}(values,realization,perm)
  end
end

function TransientSnapshots(
  values::AbstractVector{P},
  realization::TransientParamRealization,
  perm::Array{Int}
  ) where P<:ParamArray
  TransientTTSnapshots(values,realization,perm)
end

# num_space_dofs_1D(s::TransientTTSnapshots) = length(first(first(s.values)))

function tensor_getindex(s::TransientTTSnapshots,ispace,itime,iparam)
  perm_ispace = s.permutation[ispace]
  s.values[itime][iparam][perm_ispace]
end

function tensor_setindex!(s::TransientTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = s.permutation[ispace]
  s.values[itime][iparam][perm_ispace] = v
end

function BasicSnapshots(
  s::TransientTTSnapshots{T,<:ParamArray{T,N,A}}
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
  BasicSnapshots(basic_values,s.realization,s.permutation)
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
  srange = isa(spacerange,Colon) ? Base.OneTo.(num_space_dofs(s)) : spacerange
  srange = map(i->isa(i,Integer) ? [i] : i,srange)
  trange = isa(timerange,Colon) ? Base.OneTo(num_times(s)) : timerange
  trange = isa(trange,Integer) ? [trange] : trange
  prange = isa(paramrange,Colon) ? Base.OneTo(num_params(s)) : paramrange
  prange = isa(prange,Integer) ? [prange] : prange
  selected_indices = (srange,trange,prange)
  SelectedSnapshotsAtIndices(s,selected_indices)
end

space_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[1]
time_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[2]
param_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[3]
num_space_dofs(s::SelectedTTSnapshotsAtIndices) = length.(space_indices(s))
FEM.num_times(s::SelectedTTSnapshotsAtIndices) = length(time_indices(s))
FEM.num_params(s::SelectedTTSnapshotsAtIndices) = length(param_indices(s))

function tensor_getindex(s::SelectedTTSnapshotsAtIndices,ispace,itime,iparam)
  is = CartesianIndex(getindex.(space_indices(s),Tuple(ispace)))
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  getindex(s.snaps,is,it,ip)
end

function tensor_getindex(s::SelectedTTSnapshotsAtIndices,v,ispace,itime,iparam)
  is = CartesianIndex(getindex.(space_indices(s),Tuple(ispace)))
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedTTSnapshotsAtIndices{T,N,<:BasicTTSnapshots}) where {T,N}
  v = get_values(s.snaps)
  array = Vector{typeof(first(v))}(undef,num_cols(s))
  ispace = space_indices(s)
  perm = get_permutation(s)
  perm_ispace = map(i->perm[CartesianIndex(i)],ispace)
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
  p = get_permutation(s)
  BasicTTSnapshots(values,r,p)
end

function BasicSnapshots(s::SelectedTTSnapshotsAtIndices{T,N,<:TransientTTSnapshots}) where {T,N}
  v = s.snaps.values
  basic_values = Vector{typeof(first(first(v)))}(undef,num_times(s)*num_params(s))
  ispace = space_indices(s)
  perm = get_permutation(s)
  perm_ispace = map(i->perm[CartesianIndex(i)],ispace)
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      basic_values[(i-1)*num_params(s)+j] = v[it][jp][perm_ispace]
    end
  end
  r = get_realization(s)
  BasicTTSnapshots(ParamArray(basic_values),r,perm)
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

const BasicNnzTTSnapshots = BasicTTSnapshots{T,N,P,R} where {T,N,P<:ParamSparseMatrix,R}
const TransientNnzTTSnapshots = TransientTTSnapshots{T,N,P,R} where {T,N,P<:ParamSparseMatrix,R}
const SelectedNnzTTSnapshotsAtIndices = SelectedTTSnapshotsAtIndices{T,N,S,I} where {T,N,S<:Union{BasicNnzTTSnapshots,TransientNnzTTSnapshots},I}
const NnzTTSnapshots = Union{
  BasicNnzTTSnapshots{T,N},
  TransientNnzTTSnapshots{T,N},
  SelectedNnzTTSnapshotsAtIndices{T,N}} where {T,N}

# num_space_dofs_1D(s::BasicNnzTTSnapshots) = nnz(first(s.values))

function tensor_getindex(s::BasicNnzTTSnapshots,ispace,itime,iparam)
  perm_ispace = s.permutation[ispace]
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[perm_ispace]
end

function tensor_setindex!(s::BasicNnzTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = s.permutation[ispace]
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[perm_ispace] = v
end

# num_space_dofs_1D(s::TransientNnzTTSnapshots) = nnz(first(first(s.values)))

function tensor_getindex(s::TransientNnzTTSnapshots,ispace,itime,iparam)
  nonzeros(s.values[itime][iparam])[ispace]
end

function tensor_setindex!(s::TransientNnzTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = s.permutation[ispace]
  nonzeros(s.values[itime][iparam])[perm_ispace] = v
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

function recast(s::NnzTTSnapshots,a::AbstractArray{T,3}) where T
  @check size(a,1) == 1
  v = isa(s,BasicTTSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  m,n = size(v)
  asparse = map(eachcol(dropdims(a;dims=1))) do v
    sparse(i,j,v,m,n)
  end
  return VecOfSparseMat2Arr3(asparse)
end

struct VecOfSparseMat2Arr3{Tv,Ti,V} <: AbstractArray{Tv,3}
  values::V
  function VecOfSparseMat2Arr3(values::V) where {Tv,Ti,V<:AbstractVector{<:AbstractSparseMatrix{Tv,Ti}}}
    new{Tv,Ti,V}(values)
  end
end

FEM.get_values(s::VecOfSparseMat2Arr3) = s.values
Base.size(s::VecOfSparseMat2Arr3) = (1,nnz(first(s.values)),length(s.values))

function Base.getindex(s::VecOfSparseMat2Arr3,i::Integer,j,k::Integer)
  @check i == 1
  nonzeros(s.values[k])[j]
end

function Base.getindex(s::VecOfSparseMat2Arr3,i::Integer,j,k)
  @check i == 1
  view(s,i,j,k)
end

function get_nonzero_indices(s::VecOfSparseMat2Arr3)
  get_nonzero_indices(first(s.values))
end
