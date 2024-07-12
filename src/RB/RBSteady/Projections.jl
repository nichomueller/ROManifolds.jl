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

function Projection(s::UnfoldingSteadySnapshots,args...;kwargs...)
  basis = tpod(s,args...;kwargs...)
  basis′ = recast(s,basis)
  PODBasis(basis′)
end

function Projection(s::AbstractSteadySnapshots,args...;kwargs...)
  cores = ttsvd(s,args...;kwargs...)
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

SteadyProjection stemming from a truncated proper orthogonal decomposition [`tpod`](@ref)

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

const FixedDofsTTSVDCores{D,T,A<:AbstractVector{<:AbstractArray{T,D}},I<:FixedDofsIndexMap} = TTSVDCores{D,T,A,I}

IndexMaps.get_index_map(a::TTSVDCores) = a.index_map

get_cores(a::TTSVDCores) = a.cores
get_cores_space(a::TTSVDCores) = a.cores

get_basis_space(a::TTSVDCores) = cores2basis(get_index_map(a),get_cores_space(a)...)
num_space_dofs(a::TTSVDCores) = prod(_num_tot_space_dofs(a))
function num_space_dofs(a::FixedDofsTTSVDCores)
  prod(_num_tot_space_dofs(a)) - length(get_fixed_dofs(get_index_map(a)))
end
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

function compress_cores(core::TTSVDCores,bases::TTSVDCores...;kwargs...)
  ccores = map((a,b...)->compress_core(a,b...;kwargs...),get_cores(core),get_cores.(bases)...)
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
  cores = [get_cores_space(a[i]) for i = get_touched_blocks(a)]
  return return_cache(block_map,cores...)
end

function get_cores(a::BlockProjection)
  active_block_ids = get_touched_blocks(a)
  block_map = BlockMap(size(a),active_block_ids)
  cores = [get_cores(a[i]) for i = get_touched_blocks(a)]
  return return_cache(block_map,cores...)
end

function IndexMaps.get_index_map(a::BlockProjection)
  active_block_ids = get_touched_blocks(a)
  index_map = [get_index_map(a[i]) for i = get_touched_blocks(a)]
  return index_map
end

function Projection(s::BlockSnapshots;kwargs...)
  norm_matrix = fill(nothing,size(s))
  Projection(s,norm_matrix;kwargs...)
end

function Projection(s::BlockSnapshots,norm_matrix;kwargs...)
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  bases = [Projection(s[i],norm_matrix[Block(i,i)];kwargs...) for i in active_block_ids]
  BlockProjection(block_map,bases)
end

"""
    enrich_basis(
      a::BlockProjection,
      norm_matrix::AbstractMatrix,
      supr_op::AbstractMatrix) -> BlockProjection

Returns the supremizer-enriched BlockProjection. This function stabilizes Inf-Sup
problems projected on a reduced vector space

"""
function enrich_basis(a::BlockProjection{<:PODBasis},norm_matrix::AbstractMatrix,supr_op::AbstractMatrix)
  bases = add_space_supremizers(get_basis_space(a),norm_matrix,supr_op)
  return BlockProjection(map(PODBasis,bases),a.touched)
end

function enrich_basis(a::BlockProjection{<:TTSVDCores},norm_matrix::AbstractMatrix,supr_op::AbstractMatrix)
  cores = add_tt_supremizers(get_cores_space(a),norm_matrix,supr_op)
  return BlockProjection(map(TTSVDCores,cores),a.touched)
end

"""
    add_space_supremizers(
      basis_space::MatrixBlock,
      norm_matrix::AbstractMatrix,
      supr_op::AbstractMatrix) -> Vector{<:AbstractArray}

Enriches the spatial basis with spatial supremizers computed from
the action of the supremizing operator `supr_op` on the dual field(s)

"""
function add_space_supremizers(basis_space::MatrixBlock,norm_matrix::AbstractMatrix,supr_op::AbstractMatrix)
  basis_primal,basis_dual... = basis_space.array
  A = norm_matrix[Block(1,1)]
  H = cholesky(A)
  for i = eachindex(basis_dual)
    C = supr_op[Block(1,i+1)]
    b_i = C * basis_dual[i]
    supr_i = H \ b_i
    gram_schmidt!(supr_i,basis_primal,A)
    basis_primal = hcat(basis_primal,supr_i)
  end
  return [basis_primal,basis_dual...]
end

function add_tt_supremizers(cores_space::MatrixBlock,norm_matrix::BlockTProductArray,supr_op::BlockTProductArray)
  pblocks,dblocks = TProduct.primal_dual_blocks(supr_op)
  cores_primal′ = BlockVectorTTCores()

  for id in dblocks
    rcores = Array{T,3}[]
    rcore = Matrix{T}[]
    C = supr_op[Block(ip,id)]
    for ip in pblocks
      A = norm_matrix[Block(ip,ip)]
      reduced_coupling!((rcores,rcore),cores_space[ip],cores_space[id],A,C)
    end
    cores[ip] = enrich(cores_space[ip],rcores,vcat(rcore...))
  end

  return cores
end

function reduced_coupling!(cache,cores_primal_i,cores_dual_i,norm_matrix_i,coupling_i)
  rcores_dual,rcore = cache
  # cores_norm_i = TProduct.tp_decomposition(norm_matrix_i) # add here the norm matrix
  cores_coupling_i = TProduct.tp_decomposition(coupling_i)
  rcores_dual_i = _compress1.(cores_coupling_i,cores_dual_i) # add here the norm matrix
  rcores_i = compress2.(cores_primal_i,rcores_dual_i)
  rcore_i = multiply_cores(rcores_i...) |> _dropdims
  push!(rcores_dual,rcores_dual_i)
  push!(rcore,rcore_i)
end

function _coupling_kron_dual(B::Vector{<:AbstractSparseMatrix{T}},cores::Vector{<:AbstractArray{S}}) where {T,S}

end

function _primal_kron_rcoupling(B::Vector{<:AbstractSparseMatrix},)

end

function enrich(basis_primal,basis_dual;tol=1e-2)
  basis_pd = basis_primal'*basis_dual

  function _enrich(basis_primal,basis_pd,v)
    vnew = copy(v)
    orth_complement!(vnew,basis_primal)
    vnew /= norm(vnew)
    hcat(basis_primal,vnew),vcat(basis_pd,vnew'*basis_dual)
  end

  for i = size(basis_pd,2) - size(basis_pd,1)
    basis_primal,basis_pd = _enrich(basis_primal,basis_pd,basis_dual[:,i])
  end

  i = 1
  while i ≤ size(basis_pd,2)
    proj = i == 1 ? zeros(size(basis_pd,1)) : orth_projection(basis_pd[:,i],basis_pd[:,1:i-1])
    dist = norm(basis_pd[:,1]-proj)
    if dist ≤ tol
      basis_primal,basis_pd = _enrich(basis_primal,basis_pd,basis_dual[:,i])
      i = 0
    else
      basis_pd[:,i] .-= proj
    end
    i += 1
  end

  return basis_primal
end
