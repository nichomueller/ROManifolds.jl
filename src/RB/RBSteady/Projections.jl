"""
    abstract type Projection end

Represents a basis for a vector (sub)space, used as a (Petrov)-Galerkin projection
operator. In other words, Projection variables are operators from a high dimensional
vector space to a low dimensional one

Subtypes:
- [`SteadyProjection`](@ref)
- [`TransientProjection`](@ref)
- [`ReducedAlgebraicOperator`](@ref)

"""
abstract type Projection end

get_basis_space(a::Projection) = @abstractmethod
num_space_dofs(a::Projection) = @abstractmethod
num_reduced_space_dofs(a::Projection) = @abstractmethod
num_fe_dofs(a::Projection) = @abstractmethod
num_reduced_dofs(a::Projection) = @abstractmethod

"""
    abstract type SteadyProjection <: Projection end

Specialization for projections in steady problems. A constructor is given by
the function Projection

Subtypes:
- [`PODBasis`](@ref)
- [`TTSVDCores`](@ref)

"""
abstract type SteadyProjection <: Projection end

num_fe_dofs(a::SteadyProjection) = num_space_dofs(a)
num_reduced_dofs(a::SteadyProjection) = num_reduced_space_dofs(a)

function Projection(red::PODReduction,s::AbstractSteadySnapshots,args...)
  basis = projection(red,s,args...)
  basis′ = recast(s,basis)
  PODBasis(basis′)
end

function Projection(red::TTSVDReduction,s::AbstractSteadySnapshots,args...)
  cores = projection(red,s,args...)
  cores′ = recast(s,cores)
  index_map = get_index_map(s)
  TTSVDCores(cores′,index_map)
end

"""
    recast(x̂::AbstractVector,a::Projection) -> AbstractVector

Returns the action of the transposed Projection operator `a` on `x̂`

"""
function IndexMaps.recast(x̂::AbstractVector,a::SteadyProjection)
  basis = get_basis_space(a)
  x = basis*x̂
  return x
end

"""
    struct PODBasis{A<:AbstractMatrix} <: SteadyProjection

SteadyProjection stemming from a truncated proper orthogonal decomposition [`truncated_pod`](@ref)

"""
struct PODBasis{A<:AbstractMatrix} <: SteadyProjection
  basis::A
end

get_basis_space(a::PODBasis) = a.basis
num_space_dofs(a::PODBasis) = size(get_basis_space(a),1)
num_reduced_space_dofs(a::PODBasis) = size(get_basis_space(a),2)

# TT interface

"""
    TTSVDCores{D,T,A<:AbstractVector{<:AbstractArray{T,D}},I} <: SteadyProjection

SteadyProjection stemming from a tensor train singular value decomposition [`ttsvd`](@ref).
An index map of type `I` is provided for indexing purposes

"""
struct TTSVDCores{D,T,A<:AbstractVector{<:AbstractArray{T,D}},I} <: SteadyProjection
  cores::A
  index_map::I
end

IndexMaps.get_index_map(a::TTSVDCores) = a.index_map

get_cores(a::TTSVDCores) = a.cores
get_cores_space(a::TTSVDCores) = a.cores

get_basis_space(a::TTSVDCores) = cores2basis(get_index_map(a),get_cores_space(a)...)
num_space_dofs(a::TTSVDCores) = prod(_num_tot_space_dofs(a))
num_reduced_space_dofs(a::TTSVDCores) = size(last(get_cores_space(a)),3)

_num_tot_space_dofs(a::TTSVDCores{3}) = size.(get_cores_space(a),2)

function _num_tot_space_dofs(a::TTSVDCores{4})
  scores = get_cores_space(a)
  tot_ndofs = zeros(Int,2,length(scores))
  @inbounds for i = eachindex(scores)
    tot_ndofs[:,i] .= size(scores[i],2),size(scores[i],3)
  end
  return tot_ndofs
end

function compress_cores(core::TTSVDCores,basis_test::TTSVDCores)
  ccores = map((a,btest)->compress_core(a,btest),get_cores(core),get_cores(basis_test))
  ccore = multiply_cores(ccores...)
  _dropdims(ccore)
end

function compress_cores(core::TTSVDCores,basis_trial::TTSVDCores,bases::TTSVDCores)
  ccores = map((a,btrial,btest)->compress_core(a,btrial,btest),get_cores(core),
    get_cores(basis_trial),get_cores(basis_test))
  ccore = multiply_cores(ccores...)
  _dropdims(ccore)
end

# multi field interface

"""
    struct BlockProjection{A,N} <: AbstractArray{Projection,N} end

Block container for Projection of type `A` in a MultiField setting. This
type is conceived similarly to [`ArrayBlock`](@ref) in [`Gridap`](@ref)

"""
struct BlockProjection{A,N} <: AbstractArray{Projection,N}
  array::Array{A,N}
  touched::Array{Bool,N}

  function BlockProjection(
    array::Array{A,N},
    touched::Array{Bool,N}
    ) where {A<:Projection,N}

    @check size(array) == size(touched)
    new{A,N}(array,touched)
  end
end

function BlockProjection(k::BlockMap{N},a::AbstractArray{A}) where {A<:Projection,N}
  array = Array{A,N}(undef,k.size)
  touched = fill(false,k.size)
  for (t,i) in enumerate(k.indices)
    array[i] = a[t]
    touched[i] = true
  end
  BlockProjection(array,touched)
end

Base.size(a::BlockProjection,i...) = size(a.array,i...)

function Base.getindex(a::BlockProjection,i...)
  if !a.touched[i...]
    return nothing
  end
  a.array[i...]
end

function Base.setindex!(a::BlockProjection,v,i...)
  @check a.touched[i...] "Only touched entries can be set"
  a.array[i...] = v
end

function get_touched_blocks(a::BlockProjection)
  findall(a.touched)
end

function get_basis_space(a::BlockProjection{A,N}) where {A,N}
  basis_space = Array{Matrix{Float64},N}(undef,size(a))
  touched = a.touched
  for i in eachindex(a)
    if touched[i]
      basis_space[i] = get_basis_space(a[i])
    end
  end
  return ArrayBlock(basis_space,a.touched)
end

function num_space_dofs(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_space_dofs(a[i])
    end
  end
  return dofs
end

function num_reduced_space_dofs(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_reduced_space_dofs(a[i])
    end
  end
  return dofs
end

function get_cores_space(a::BlockProjection)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  cores = [get_cores_space(a[i]) for i = active_block_ids]
  return return_cache(block_map,cores...)
end

function get_cores(a::BlockProjection)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  cores = [get_cores(a[i]) for i = active_block_ids]
  return return_cache(block_map,cores...)
end

function IndexMaps.get_index_map(a::BlockProjection)
  active_block_ids = get_touched_blocks(a)
  index_map = [get_index_map(a[i]) for i = active_block_ids]
  return index_map
end

function Projection(red::AbstractReduction,s::BlockSnapshots)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  bases = [Projection(red,s[i]) for i in active_block_ids]
  BlockProjection(block_map,bases)
end

function Projection(red::AbstractReduction,s::BlockSnapshots,norm_matrix)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  bases = [Projection(red,s[i],norm_matrix[Block(i,i)]) for i in active_block_ids]
  BlockProjection(block_map,bases)
end

"""
    enrich_basis(
      a::BlockProjection,
      norm_matrix::AbstractMatrix,
      supr_matrix::AbstractMatrix,
      args...) -> BlockProjection

Returns the supremizer-enriched BlockProjection. This function stabilizes Inf-Sup
problems projected on a reduced vector space

"""
function enrich_basis(a::BlockProjection{<:PODBasis},args...)
  bases = add_space_supremizers(get_basis_space(a),args...)
  return BlockProjection(map(PODBasis,bases),a.touched)
end

function enrich_basis(a::BlockProjection{<:TTSVDCores},args...)
  cores = add_tt_supremizers(get_cores_space(a),args...)
  return BlockProjection(map(TTSVDCores,cores),a.touched)
end

"""
    add_space_supremizers(
      basis_space::ArrayBlock,
      norm_matrix::AbstractMatrix,
      supr_matrix::AbstractMatrix) -> Vector{<:AbstractArray}

Enriches the spatial basis with spatial supremizers computed from
the action of the supremizing matrix `supr_matrix` on the dual field(s)

"""
function add_space_supremizers(basis_space::ArrayBlock,norm_matrix::AbstractMatrix,supr_matrix::AbstractMatrix)
  basis_primal,basis_dual... = basis_space.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = cholesky(X_primal)
  for i = eachindex(basis_dual)
    C_primal_dual_i = supr_matrix[Block(1,i+1)]
    supr_i = H_primal \ C_primal_dual_i * basis_dual[i]
    gram_schmidt!(supr_i,basis_primal,X_primal)
    basis_primal = hcat(basis_primal,supr_i)
  end
  return [basis_primal,basis_dual...]
end

function add_tt_supremizers(cores_space::ArrayBlock,norm_matrix::BlockGenericRankTensor,supr_op::BlockGenericRankTensor)
  pblocks,dblocks = TProduct.primal_dual_blocks(supr_op)
  cores_primal = map(ip -> cores_space[ip],pblocks)
  cores_dual = map(id -> cores_space[id],dblocks)
  norms_primal = map(ip -> norm_matrix[Block(ip,ip)],pblocks)

  for id in dblocks
    rcores = Vector{Array{Float64,3}}[]
    rcore = Matrix{Float64}[]
    cores_dual_i = cores_space[id]
    for ip in eachindex(pblocks)
      A = norm_matrix[Block(ip,ip)]
      C = supr_op[Block(ip,id)]
      cores_primal_i = cores_space[ip]
      reduced_coupling!((rcores,rcore),cores_primal_i,cores_dual_i,A,C)
    end
    enrich!(cores_primal,rcores,vcat(rcore...),norms_primal)
  end

  cores_primal,cores_dual
end

function reduced_coupling!(cache,cores_primal_i,cores_dual_i,norm_matrix_i,coupling_i)
  rcores_dual,rcore = cache
  # cores_norm_i = TProduct.tp_decomposition(norm_matrix_i) # add here the norm matrix
  cores_coupling_i = TProduct.tp_decomposition(coupling_i)
  rcores_dual_i,rcores_i = map(cores_primal_i,cores_dual_i,cores_coupling_i) do cp,cd,cc
    rc = cc*cd
    rc,compress_core(rc,cp)
  end |> tuple_of_arrays
  rcore_i = multiply_cores(rcores_i...) |> _dropdims
  push!(rcores_dual,rcores_dual_i)
  push!(rcore,rcore_i)
end

function enrich!(cores_primal,rcores,rcore,norms_primal;tol=1e-2)
  @check length(cores_primal) == length(rcores)

  flag = false
  i = 1
  while i ≤ size(rcore,2)
    proj = i == 1 ? zeros(size(rcore,1)) : orth_projection(rcore[:,i],rcore[:,1:i-1])
    dist = norm(rcore[:,i]-proj)
    if dist ≤ tol
      for ip in eachindex(cores_primal)
        cp,rc,np = cores_primal[ip],rcores[ip],norms_primal[ip]
        cores_primal[ip] = add_and_orthogonalize(cp,rc,np,i;flag)
      end
      rcore = _update_reduced_coupling(cores_primal,rcores,rcore)
      flag = true
    end
    i += 1
  end
end

function add_and_orthogonalize(cores_primal,rcores,norms_primal,i;flag=false)
  D = length(cores_primal)
  weights = Vector{Array{Float64,3}}(undef,D-1)

  if !flag
    for d in 1:D-1
      push!(cores_primal[d],rcores[d])
      _weight_array!(weights,cores_primal,norms_primal,Val{d}())
    end
    push!(cores_primal[D],rcores[D])
    _ = orthogonalize!(cores_primal[D],norms_primal,weights)
  else
    cores_primal[D] = hcat(cores_primal[D],rcores[D][:,:,i])
    _ = orthogonalize!(cores_primal[D],norms_primal,weights)
  end
end

function _update_reduced_coupling(cores_primal,rcores,rcore)
  offsets = size.(last.(cores_primal),3)
  @check size(rcore,1) == sum(offsets)
  enriched_offsets = offsets .+ 1
  pushfirst!(0,offsets)
  pushfirst!(0,enriched_offsets)
  rcore_new = similar(rcore,sum(enriched_offsets),size(rcore,2))
  for (ip,cores_primal_i) in enumerate(cores_primal)
    range = offsets[ip]+1:offsets[ip+1]
    enriched_range = enriched_offsets[ip]+1:enriched_offsets[ip+1]-1
    @views rcore_new[enriched_range,:] = rcore[range,:]
    rcores_i = compress_core.(rcores[ip],blocks(cores_primal_i)[end])
    rcore_i = multiply_cores(rcores_i...) |> _dropdims
    @views rcore_new[enriched_offsets[ip+1],:] = rcore_i
  end
  return rcore_new
end
