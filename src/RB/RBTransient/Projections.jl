"""
    abstract type TransientProjection <: Projection end

Specialization for projections in transient problems. A constructor is given by
the function Projection

Subtypes:
- [`TransientPODBasis`](@ref)
- [`TransientTTSVDCores`](@ref)

"""
abstract type TransientProjection <: Projection end

get_basis_time(a::TransientProjection) = @abstractmethod
ParamDataStructures.num_times(a::TransientProjection) = @abstractmethod
num_reduced_times(a::TransientProjection) = @abstractmethod

RBSteady.num_fe_dofs(a::TransientProjection) = num_space_dofs(a)*num_times(a)
RBSteady.num_reduced_dofs(a::TransientProjection) = RBSteady.num_reduced_space_dofs(a)*num_reduced_times(a)

function RBSteady.Projection(s::UnfoldingTransientSnapshots,args...;kwargs...)
  s′ = flatten_snapshots(s)
  basis_space = tpod(s′,args...;kwargs...)
  basis_space′ = recast(s,basis_space)
  compressed_s2 = compress(s′,basis_space,args...;swap_mode=true)
  basis_time = tpod(compressed_s2;kwargs...)
  TransientPODBasis(basis_space′,basis_time)
end

function RBSteady.Projection(s::AbstractTransientSnapshots,args...;kwargs...)
  cores_space...,core_time = ttsvd(s,args...;kwargs...)
  cores_space′ = recast(s,cores_space)
  index_map = get_index_map(s)
  TransientTTSVDCores(cores_space′,core_time,index_map)
end

"""
    TransientPODBasis{A<:AbstractMatrix,B<:AbstractMatrix} <: TransientProjection

TransientProjection stemming from a truncated proper orthogonal decomposition [`tpod`](@ref)

"""
struct TransientPODBasis{A<:AbstractMatrix,B<:AbstractMatrix} <: TransientProjection
  basis_space::A
  basis_time::B
end

RBSteady.get_basis_space(a::TransientPODBasis) = a.basis_space
RBSteady.num_space_dofs(a::TransientPODBasis) = size(get_basis_space(a),1)
RBSteady.num_reduced_space_dofs(a::TransientPODBasis) = size(get_basis_space(a),2)

get_basis_time(a::TransientPODBasis) = a.basis_time
ParamDataStructures.num_times(a::TransientPODBasis) = size(get_basis_time(a),1)
num_reduced_times(a::TransientPODBasis) = size(get_basis_time(a),2)

function IndexMaps.recast(x̂::AbstractVector,a::TransientPODBasis)
  basis_space = get_basis_space(a)
  basis_time = get_basis_time(a)
  ns = RBSteady.num_reduced_space_dofs(a)
  nt = num_reduced_times(a)

  X̂ = reshape(x̂,ns,nt)
  X = (basis_space*X̂)*basis_time'
  return X
end

# TT interface

"""
    TransientTTSVDCores{D,T,A<:AbstractVector{<:AbstractArray{T,D}},B<:AbstractArray{T,3},
      C<:AbstractMatrix,I} <: TransientProjection

TransientProjection stemming from a tensor train singular value decomposition [`ttsvd`](@ref).
An index map of type `I` is provided for indexing purposes

"""
struct TransientTTSVDCores{D,T,A<:AbstractVector{<:AbstractArray{T,D}},B<:AbstractArray{T,3},C<:AbstractMatrix,I} <: TransientProjection
  cores_space::A
  core_time::B
  basis_spacetime::C
  index_map::I
end

function TransientTTSVDCores(
  cores_space::Vector{<:AbstractArray},
  core_time::AbstractArray,
  index_map::AbstractIndexMap)

  basis_spacetime = get_basis_spacetime(index_map,cores_space,core_time)
  TransientTTSVDCores(cores_space,core_time,basis_spacetime,index_map)
end

IndexMaps.get_index_map(a::TransientTTSVDCores) = a.index_map

RBSteady.get_cores(a::TransientTTSVDCores) = (get_spatial_cores(a)...,get_temporal_core(a))
RBSteady.get_spatial_cores(a::TransientTTSVDCores) = a.cores_space
get_temporal_core(a::TransientTTSVDCores) = a.core_time

RBSteady.get_basis_space(a::TransientTTSVDCores) = cores2basis(get_index_map(a),get_spatial_cores(a)...)
RBSteady.num_space_dofs(a::TransientTTSVDCores) = prod(_num_tot_space_dofs(a))
RBSteady.num_reduced_space_dofs(a::TransientTTSVDCores) = size(last(get_spatial_cores(a)),3)

get_basis_time(a::TransientTTSVDCores) = cores2basis(get_temporal_core(a))
ParamDataStructures.num_times(a::TransientTTSVDCores) = size(get_temporal_core(a),2)
num_reduced_times(a::TransientTTSVDCores) = size(get_temporal_core(a),3)

RBSteady.num_reduced_dofs(a::TransientTTSVDCores) = num_reduced_times(a)

get_basis_spacetime(a::TransientTTSVDCores) = a.basis_spacetime

_num_tot_space_dofs(a::TransientTTSVDCores{3}) = size.(get_spatial_cores(a),2)

const TransientTTSVDSparseCores{T,A<:AbstractVector{<:SparseCore{T}},B,C} = TransientTTSVDCores{4,T,A,B,C}

function _num_tot_space_dofs(a::TransientTTSVDSparseCores)
  scores = get_spatial_cores(a)
  tot_ndofs = zeros(Int,2,length(scores))
  @inbounds for i = eachindex(scores)
    tot_ndofs[:,i] .= size(scores[i],2),size(scores[i],3)
  end
  return tot_ndofs
end

const TransientMultiValueTTSVDSparseCores{T,A<:AbstractVector{<:AbstractTTCore{T}},B,C} = TransientTTSVDCores{4,T,A,B,C}

function _num_tot_space_dofs(a::TransientMultiValueTTSVDSparseCores)
  scores = get_spatial_cores(a)
  tot_ndofs = zeros(Int,2,length(scores))
  @inbounds for i = eachindex(scores)
    tot_ndofs[:,i] .= size(scores[i],2),size(scores[i],3)
  end
  return tot_ndofs
end

function RBSteady.basis2cores(mat::AbstractMatrix,a::TransientTTSVDCores)
  index_map = get_index_map(a)
  D = ndims(index_map)
  s = reverse(size(index_map))
  v = reshape(mat,num_times(a),s...,:)
  permutedims(v,(reverse(1:D+1)...,D+2))
end

function get_basis_spacetime(index_map::AbstractIndexMap,cores_space,core_time)
  cores2basis(RBSteady._cores2basis(index_map,cores_space...),core_time)
end

function IndexMaps.recast(x̂::AbstractVector,a::TransientTTSVDCores)
  basis_spacetime = get_basis_spacetime(a)
  Ns = num_space_dofs(a)
  Nt = num_times(a)

  x = basis_spacetime*x̂
  X = reshape(x,Ns,Nt)
  return X
end

# multi field interface

function get_basis_time(a::BlockProjection{A,N}) where {A,N}
  basis_time = Array{Matrix{Float64},N}(undef,size(a))
  touched = a.touched
  for i in eachindex(a)
    if touched[i]
      basis_time[i] = get_basis_time(a[i])
    end
  end
  return ArrayBlock(basis_time,a.touched)
end

ParamDataStructures.num_times(a::BlockProjection) = num_times(a[findfirst(a.touched)])

function num_reduced_times(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_reduced_times(a[i])
    end
  end
  return dofs
end

function get_temporal_core(a::BlockProjection)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  cores = [get_temporal_core(a[i]) for i = get_touched_blocks(a)]
  return return_cache(block_map,cores...)
end

function RBSteady.enrich_basis(
  a::BlockProjection{<:TransientPODBasis},
  norm_matrix::AbstractMatrix,
  supr_op::AbstractMatrix)

  basis_space = add_space_supremizers(a,norm_matrix,supr_op)
  basis_time = add_time_supremizers(a)
  basis = BlockProjection(map(TransientPODBasis,basis_space,basis_time),a.touched)
  return basis
end

function RBSteady.enrich_basis(
  a::BlockProjection{<:TransientTTSVDCores},
  norm_matrix::AbstractMatrix,
  supr_op::AbstractMatrix)

  bs_primal,bs_dual... = add_space_supremizers(a,norm_matrix,supr_op)
  bt_primal,bt_dual... = add_time_supremizers(a)
  bst_primal = kronecker(bt_primal,bs_primal)
  snaps_primal = basis2cores(bst_primal,a[1,1])
  cores_space_primal...,core_time_primal = ttsvd(snaps_primal,norm_matrix[1,1];ϵ=1e-10)
  _,cores_space_dual... = get_cores_space(a).array
  _,core_time_dual... = get_cores_time(a).array
  cores_space = (cores_space_primal,cores_space_dual...)
  cores_time = (core_time_primal,core_time_dual...)
  basis = BlockProjection(map(TransientTTSVDCores,cores_space,cores_time),a.touched)
  return basis
end

"""
    add_time_supremizers(a::BlockProjection{<:TransientProjection};kwargs...) -> Vector{<:Matrix}

Enriches the temporal basis with temporal supremizers computed from
the kernel of the temporal basis associated to the primal field projected onto
the column space of the temporal basis (bases) associated to the duel field(s)

"""
function add_time_supremizers(a::BlockProjection;kwargs...)
  basis_time = get_basis_time(a)
  basis_primal,basis_dual... = basis_time.array
  for i = eachindex(basis_dual)
    basis_primal = add_time_supremizers(basis_primal,basis_dual[i];kwargs...)
  end
  return [basis_primal,basis_dual...]
end

function add_time_supremizers(basis_primal,basis_dual;tol=1e-2)
  basis_pd = basis_primal'*basis_dual

  function enrich(basis_primal,basis_pd,v)
    vnew = copy(v)
    orth_complement!(vnew,basis_primal)
    vnew /= norm(vnew)
    hcat(basis_primal,vnew),vcat(basis_pd,vnew'*basis_dual)
  end

  count = 0
  ntd_minus_ntp = size(basis_dual,2) - size(basis_primal,2)
  if ntd_minus_ntp > 0
    for ntd = 1:ntd_minus_ntp
      basis_primal,basis_pd = enrich(basis_primal,basis_pd,basis_dual[:,ntd])
      count += 1
    end
  end

  ntd = 1
  while ntd ≤ size(basis_pd,2)
    proj = ntd == 1 ? zeros(size(basis_pd,1)) : orth_projection(basis_pd[:,ntd],basis_pd[:,1:ntd-1])
    dist = norm(basis_pd[:,1]-proj)
    if dist ≤ tol
      basis_primal,basis_pd = enrich(basis_primal,basis_pd,basis_dual[:,ntd])
      count += 1
      ntd = 0
    else
      basis_pd[:,ntd] .-= proj
    end
    ntd += 1
  end

  return basis_primal
end
