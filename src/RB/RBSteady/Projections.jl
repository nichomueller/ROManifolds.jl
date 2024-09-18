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
abstract type Projection <: Map end

struct InvProjection{P} <: Projection
  projection::P
end

Base.adjoint(a::Projection) = InvProjection(a)

Base.:*(a::InvProjection,x::AbstractArray) = inv_project(a.projection,x)
Base.:*(a::Projection,x::AbstractArray) = project(a,x)

function project(a::Projection,x::AbstractArray)
  basis = get_basis(a)
  x̂ = basis'*x
  return x̂
end

function inv_project(a::Projection,x̂::AbstractArray)
  basis = get_basis(a)
  x = basis*x̂
  return x
end

get_basis(a::Projection) = @abstractmethod
num_fe_dofs(a::Projection) = @abstractmethod
num_reduced_dofs(a::Projection) = @abstractmethod

function projection(red::AbstractReduction,s::AbstractSnapshots,args...)
  basis = reduction(red,s,args...)
  index_map = get_index_map(s)
  projection(red,basis,index_map)
end

"""
    abstract type SteadyProjection <: Projection end

Specialization for projections in steady problems. A constructor is given by
the function Projection

Subtypes:
- [`PODBasis`](@ref)

"""
abstract type SteadyProjection <: Projection end

"""
    struct PODBasis{A<:AbstractMatrix} <: SteadyProjection

SteadyProjection stemming from a truncated proper orthogonal decomposition [`truncated_pod`](@ref)

"""
struct PODBasis{A<:AbstractMatrix,I<:AbstractIndexMap} <: SteadyProjection
  basis::A
  index_map::I
end

function projection(::PODReduction,basis::AbstractMatrix,index_map::AbstractIndexMap)
  PODBasis(basis,index_map)
end

get_basis(a::PODBasis) = a.basis
num_fe_dofs(a::PODBasis) = size(get_basis(a),1)
num_reduced_dofs(a::PODBasis) = size(get_basis(a),2)

IndexMaps.get_index_map(a::PODBasis) = a.index_map
IndexMaps.recast(a::PODBasis) = recast(get_basis(a),get_index_map(a))

# TT interface

"""
    TTSVDCores{A<:AbstractVector{<:AbstractArray{T,D}},I} <: Projection

SteadyProjection stemming from a tensor train singular value decomposition [`ttsvd`](@ref).
An index map of type `I` is provided for indexing purposes

"""
struct TTSVDCores{D,A<:AbstractVector{<:AbstractArray{T,3} where T},I<:AbstractIndexMap{D}} <: Projection
  cores::A
  index_map::I
end

function projection(red::TTSVDReduction,s::AbstractSnapshots,args...)
  cores = reduction(red,s,args...)
  index_map = get_index_map(s)
  TTSVDCores(cores,index_map)
end

get_cores(a::TTSVDCores) = a.cores

get_basis(a::TTSVDCores) = cores2basis(get_cores(a)...)
num_fe_dofs(a::TTSVDCores) = prod(map(c -> size(c,2),get_cores(a)))
num_reduced_dofs(a::TTSVDCores) = size(last(get_cores(a)),3)

IndexMaps.get_index_map(a::TTSVDCores) = a.index_map
IndexMaps.recast(a::TTSVDCores{D}) where D = recast(get_cores(a)[1:D],get_index_map(a))

# multi field interface

function Arrays.return_cache(::typeof(projection),::PODReduction,s::AbstractSnapshots)
  b = testvalue(Matrix{eltype(s)})
  i = get_index_map(s)
  return PODBasis(b,i)
end

function Arrays.return_cache(::typeof(projection),::TTSVDReduction,s::AbstractSnapshots)
  c = testvalue(Vector{Array{eltype(s),3}})
  i = get_index_map(s)
  return TTSVDCores(c,i)
end

function Arrays.return_cache(::typeof(projection),red::AbstractReduction,s::BlockSnapshots)
  basis = return_cache(projection,red,blocks(s)[1])
  touched = s.touched
  block_basis = Array{typeof(basis),ndims(s)}(undef,size(s))
  return BlockProjection(block_basis)
end

function projection(red::AbstractReduction,s::BlockSnapshots)
  basis = return_cache(projection,red,s)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i])
    end
  end
  return basis
end

function projection(red::AbstractReduction,s::BlockSnapshots,norm_matrix)
  basis = return_cache(projection,red,s)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i],norm_matrix[i])
    end
  end
  return basis
end

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

function get_basis(a::BlockProjection{A,N}) where {A,N}
  basis = Array{Matrix{Float64},N}(undef,size(a))
  touched = a.touched
  for i in eachindex(a)
    if touched[i]
      basis[i] = get_basis(a[i])
    end
  end
  return ArrayBlock(basis,a.touched)
end

function num_fe_dofs(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_fe_dofs(a[i])
    end
  end
  return dofs
end

function num_reduced_dofs(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_reduced_dofs(a[i])
    end
  end
  return dofs
end

function IndexMaps.get_index_map(a::BlockProjection{A,N}) where {A,N}
  T = typeof(get_index_map(first(a)))
  index_map = Array{T,N}(undef,size(a))
  touched = a.touched
  for i in eachindex(a)
    if touched[i]
      index_map[i] = get_index_map(a[i])
    end
  end
  return index_map
end

function get_cores(a::BlockProjection{A,N}) where {A,N}
  T = typeof(get_cores(first(a)))
  cores = Array{T,N}(undef,size(a))
  touched = a.touched
  for i in eachindex(a)
    if touched[i]
      cores[i] = get_cores(a[i])
    end
  end
  return ArrayBlock(cores,a.touched)
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

function add_tt_supremizers(cores_space::ArrayBlock,norm_matrix::BlockRankTensor,supr_op::BlockRankTensor)
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
