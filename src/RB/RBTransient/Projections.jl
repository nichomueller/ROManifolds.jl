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

const TransientFixedDofsTTSVDCores{
  D,T,A<:AbstractVector{<:AbstractArray{T,D}},B<:AbstractArray{T,3},C<:AbstractMatrix,
  I<:FixedDofsIndexMap} = TransientTTSVDCores{D,T,A,B,C,I}

function TransientTTSVDCores(
  cores_space::Vector{<:AbstractArray},
  core_time::AbstractArray,
  index_map::AbstractIndexMap)

  basis_spacetime = get_basis_spacetime(index_map,cores_space,core_time)
  TransientTTSVDCores(cores_space,core_time,basis_spacetime,index_map)
end

IndexMaps.get_index_map(a::TransientTTSVDCores) = a.index_map

RBSteady.get_cores(a::TransientTTSVDCores) = [get_cores_space(a)...,get_core_time(a)]
RBSteady.get_cores_space(a::TransientTTSVDCores) = a.cores_space
get_core_time(a::TransientTTSVDCores) = a.core_time

RBSteady.get_basis_space(a::TransientTTSVDCores) = cores2basis(get_index_map(a),get_cores_space(a)...)
RBSteady.num_space_dofs(a::TransientTTSVDCores) = prod(_num_tot_space_dofs(a))
function RBSteady.num_space_dofs(a::TransientFixedDofsTTSVDCores)
  prod(_num_tot_space_dofs(a)) - length(get_fixed_dofs(get_index_map(a)))
end
get_basis_time(a::TransientTTSVDCores) = cores2basis(get_core_time(a))
ParamDataStructures.num_times(a::TransientTTSVDCores) = size(get_core_time(a),2)
num_reduced_times(a::TransientTTSVDCores) = size(get_core_time(a),3)

RBSteady.num_reduced_dofs(a::TransientTTSVDCores) = num_reduced_times(a)

get_basis_spacetime(a::TransientTTSVDCores) = a.basis_spacetime

_num_tot_space_dofs(a::TransientTTSVDCores{3}) = size.(get_cores_space(a),2)

function _num_tot_space_dofs(a::TransientTTSVDCores{4})
  scores = get_cores_space(a)
  tot_ndofs = zeros(Int,2,length(scores))
  @inbounds for i = eachindex(scores)
    tot_ndofs[:,i] .= size(scores[i],2),size(scores[i],3)
  end
  return tot_ndofs
end

function get_basis_spacetime(index_map::AbstractIndexMap,cores_space,core_time)
  cores2basis(RBSteady._cores2basis(index_map,cores_space...),core_time)
end

function RBSteady.compress_cores(core::TransientTTSVDCores,bases::TransientTTSVDCores...;kwargs...)
  ccores = map((a,b...)->compress_core(a,b...;kwargs...),get_cores(core),get_cores.(bases)...)
  ccore = multiply_cores(ccores...)
  RBSteady._dropdims(ccore)
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

function get_core_time(a::BlockProjection)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  cores = [get_core_time(a[i]) for i = get_touched_blocks(a)]
  return return_cache(block_map,cores...)
end

function RBSteady.enrich_basis(
  a::BlockProjection{<:TransientPODBasis},
  norm_matrix::AbstractMatrix,
  supr_op::AbstractMatrix)

  basis_space = add_space_supremizers(get_basis_space(a),norm_matrix,supr_op)
  basis_time = add_time_supremizers(get_basis_time(a))
  basis = BlockProjection(map(TransientPODBasis,basis_space,basis_time),a.touched)
  return basis
end

function RBSteady.enrich_basis(
  a::BlockProjection{<:TransientTTSVDCores},
  norm_matrix::AbstractMatrix,
  supr_op::AbstractMatrix)

  cores_space,core_time,enrich = add_tt_supremizers(
    get_cores_space(a),get_core_time(a),norm_matrix,supr_op)
  if enrich # we have added supremizers
    index_map = get_index_map(a)
    a′ = BlockProjection(map(TransientTTSVDCores,cores_space,core_time,index_map),a.touched)
    return a′
  else
    return a
  end
end

"""
    add_time_supremizers(basis_time::MatrixBlock;kwargs...) -> Vector{<:Matrix}

Enriches the temporal basis with temporal supremizers computed from
the kernel of the temporal basis associated to the primal field projected onto
the column space of the temporal basis (bases) associated to the duel field(s)

"""
function add_time_supremizers(basis_time::MatrixBlock;kwargs...)
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

  i = 1
  while i ≤ size(basis_pd,2)
    proj = i == 1 ? zeros(size(basis_pd,1)) : orth_projection(basis_pd[:,i],basis_pd[:,1:i-1])
    dist = norm(basis_pd[:,1]-proj)
    if dist ≤ tol
      basis_primal,basis_pd = enrich(basis_primal,basis_pd,basis_dual[:,i])
      i = 0
    else
      basis_pd[:,i] .-= proj
    end
    i += 1
  end

  return basis_primal
end

function RBSteady.add_tt_supremizers(
  cores_space::MatrixBlock,
  core_time::MatrixBlock,
  norm_matrix::BlockTProductArray,
  supr_op::BlockTProductArray)

  pblocks,dblocks = TProduct.primal_dual_blocks(supr_op)
  cores_primal_space = map(ip -> cores_space[ip],pblocks)
  core_primal_time = map(ip -> core_time[ip],pblocks)
  cores_primal_space′ = map(cores -> BlockTTCore.([cores]),cores_primal_space)
  core_primal_time′ = map(core -> BlockTTCore.([core]),core_primal_time)
  flag = false

  for id in dblocks
    rcores = Vector{Array{T,3}}[]
    rcore = Matrix{T}[]
    cores_dual_space_i = cores_space[id]
    core_dual_time_i = core_time[id]
    for ip in eachindex(pblocks)
      A = norm_matrix[Block(ip,ip)]
      C = supr_op[Block(ip,id)]
      cores_primal_space_i = cores_space[ip]
      core_primal_time_i = core_time[ip]
      reduced_coupling!((rcores,rcore),cores_primal_space_i,core_primal_time_i,
        cores_dual_space_i,core_dual_time_i,A,C)
    end
    flag = enrich!(cores_primal_space′,core_primal_time′,rcores,vcat(rcore...))
  end

  if flag
    return [cores_primal_space′...,cores_dual...]
  else
    return [cores_primal...,cores_dual...]
  end
end

function RBSteady.reduced_coupling!(
  cache,
  cores_primal_space_i,core_primal_time_i,
  cores_dual_space_i,core_dual_time_i,
  norm_matrix_i,coupling_i)

  rcores_dual,rcore = cache
  cores_coupling_i = TProduct.tp_decomposition(coupling_i)
  rcores_dual_i = [map(*,cores_coupling_i,cores_dual_space_i)...,core_dual_time_i]
  cores_primal_i = [cores_primal_space_i...,core_primal_time_i]
  rcores_i = compress_core.(rcores_dual_i,cores_primal_i)
  rcore_i = multiply_cores(rcores_i...) |> RBSteady._dropdims
  push!(rcores_dual,rcores_dual_i)
  push!(rcore,rcore_i)
end
