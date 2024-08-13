abstract type AbstractTProductTensor{A} <: AbstractVector{A} end

get_decomposition(a::AbstractTProductTensor) = @abstractmethod

abstract type AbstractRank1Tensor{A} <: AbstractTProductTensor{A} end

get_factors(a::AbstractRank1Tensor) = @abstractmethod
get_decomposition(a::AbstractRank1Tensor) = get_factors(a)
Base.size(a::AbstractRank1Tensor) = (length(get_factors(a)),)
Base.getindex(a::AbstractRank1Tensor,d::Integer) = get_factors(a)[d]

struct GenericRank1Tensor{A} <: AbstractRank1Tensor{A}
  factors::Vector{A}
end

get_factors(a::GenericRank1Tensor) = a.factors

abstract type AbstractRankTensor{A,K} <: AbstractTProductTensor{AbstractRank1Tensor{A}} end

LinearAlgebra.rank(a::AbstractRankTensor{A,K} where A) where K = K
Base.size(a::AbstractRankTensor) = (rank(a),)
Base.getindex(a::AbstractRankTensor,k::Integer) = get_decomposition(a,k)
get_decomposition(a::AbstractRankTensor) = ntuple(k -> getindex(a,k),Val{length(a)}())

struct GenericRankTensor{A,K} <: AbstractRankTensor{A,K}
  decompositions::Vector{GenericRank1Tensor{A}}
  function GenericRankTensor(decompositions::Vector{GenericRank1Tensor{A}}) where A
    K = length(decompositions)
    new{A,K}(decompositions)
  end
end

get_decomposition(a::GenericRankTensor,k::Integer) = a.decompositions[k]
get_factor(a::GenericRankTensor,d::Integer,k::Integer) = get_factor(get_decomposition(a,k),d)

function _sort!(a::AbstractVector{<:AbstractVector},i::NTuple{1,<:TProductIndexMap})
  irow, = i
  irow1d = IndexMaps.get_univariate_indices(irow)
  for (i,(ai,irowi)) in enumerate(zip(a,irow1d))
    a[i] = ai[vec(irowi)]
  end
end

function _sort!(a::AbstractVector{<:AbstractMatrix},i::NTuple{2,<:TProductIndexMap})
  irow,icol = i
  irow1d = IndexMaps.get_univariate_indices(irow)
  icol1d = IndexMaps.get_univariate_indices(icol)
  for (i,(ai,irowi,icoli)) in enumerate(zip(a,irow1d,icol1d))
    a[i] = ai[vec(irowi),vec(icoli)]
  end
end

function tproduct_array(arrays_1d::Vector{<:AbstractArray},index_map)
  _sort!(arrays_1d,index_map)
  GenericRank1Tensor(arrays_1d)
end

function tproduct_array(
  ::Nothing,
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  index_map,
  summation=nothing)

  tproduct_array(arrays_1d,index_map)
end

function tproduct_array(
  ::typeof(gradient),
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  index_map,
  summation=nothing)

  _sort!(arrays_1d,index_map)
  _sort!(gradients_1d,index_map)
  decompositions = _find_decompositions(summation,arrays_1d,gradients_1d)
  GenericRankTensor(decompositions)
end

function _find_decompositions(::Nothing,arrays_1d,gradients_1d)
  @check length(arrays_1d) == length(gradients_1d)
  inds = LinearIndices(arrays_1d)
  d = map(inds) do i
    di = copy(arrays_1d)
    di[i] = gradients_1d[i]
    return GenericRank1Tensor(di)
  end
  return d
end

function _find_decompositions(summation,arrays_1d,gradients_1d)
  @check length(arrays_1d) == length(gradients_1d)
  inds = LinearIndices(arrays_1d)
  d0 = GenericRank1Tensor(arrays_1d)
  d = typeof(d0)[]
  push!(d,d0)
  map(inds) do i
    di = copy(arrays_1d)
    di[i] = gradients_1d[i]
    push!(d,GenericRank1Tensor(rank1))
  end
  return d
end

# MultiField interface
struct BlockGenericRankTensor{A<:AbstractTProductTensor,N} <: AbstractArray{A,N}
  array::Array{A,N}
end

function tproduct_array(arrays_1d::Vector{<:BlockArray},index_map)
  s_blocks = blocksize(first(arrays_1d))
  arrays = map(CartesianIndices(s_blocks)) do i
    iblock = Block(Tuple(i))
    arrays_1d_i = getindex.(arrays_1d,iblock)
    index_map_i = getindex.(index_map,Tuple(i))
    tproduct_array(arrays_1d_i,index_map_i)
  end
  BlockGenericRankTensor(arrays)
end

function tproduct_array(op::ArrayBlock,arrays_1d::Vector{<:BlockArray},gradients_1d::Vector{<:BlockArray},index_map,s::ArrayBlock)
  s_blocks = blocksize(first(arrays_1d))
  arrays = map(CartesianIndices(s_blocks)) do i
    iblock = Block(Tuple(i))
    arrays_1d_i = getindex.(arrays_1d,iblock)
    gradients_1d_i = getindex.(gradients_1d,iblock)
    index_map_i = getindex.(index_map,Tuple(i))
    op_i = op[Tuple(i)...]
    s_i = s[Tuple(i)...]
    tproduct_array(op_i,arrays_1d_i,gradients_1d_i,index_map_i,s_i)
  end
  BlockGenericRankTensor(arrays)
end

Base.size(a::BlockGenericRankTensor) = size(a.array)

function Base.getindex(
  a::BlockGenericRankTensor{A,N},
  i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  getindex(a.array,i...)
end

function Base.setindex!(
  a::BlockGenericRankTensor{A,N},
  v,i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  setindex!(a.array,v,i...)
end

function Base.getindex(a::BlockGenericRankTensor{A,N},i::Block{N}) where {A,N}
  getindex(a.array,i.n...)
end

function primal_dual_blocks(a::BlockGenericRankTensor)
  primal_blocks = Int[]
  dual_blocks = Int[]
  for i in CartesianIndices(size(a))
    if !(all(iszero.(a[i].arrays_1d)))
      irow,icol = Tuple(i)
      push!(primal_blocks,irow)
      push!(dual_blocks,icol)
    end
  end
  unique!(primal_blocks)
  unique!(dual_blocks)
  return primal_blocks,dual_blocks
end

# linear algebra

function Base.:*(a::AbstractRank1Tensor,b::AbstractArray{T,N}) where {T,N}
  @check length(a) == N
  @check all((size(a[i],2) == size(b,i)) for i = 1:N)
  c = similar(b,ntuple(i->size(a[i],1),Val{N}()))
  for i = 1:N
    bi = eachslice(b,dims=i)
    ci = eachslice(c,dims=i)
    ci .= get_factor(a,i)*bi
  end
  return c
end

function Base.:*(a::AbstractArray{T,N},b::AbstractRank1Tensor) where {T,N}
  @check length(b) == N
  @check all((size(a,i) == size(b[i],1)) for i = 1:N)
  c = similar(a,ntuple(i->size(b[i],2),Val{N}()))
  for i = 1:N
    ai = eachslice(a,dims=i)
    ci = eachslice(c,dims=i)
    ci .= ai*get_factor(b,i)
  end
  return c
end

function Base.:*(a::AbstractRankTensor,b::AbstractArray{T,N}) where {T,N}
  c = get_decomposition(a,1)*b
  for k = 2:rank(a)
    c += get_decomposition(a,k)*b
  end
  return c
end

function Base.:*(a::AbstractArray{T,N},b::AbstractRankTensor) where {T,N}
  c = a*get_decomposition(b,1)
  for k = 2:rank(b)
    c += a*get_decomposition(b,k)
  end
  return c
end

# to global array - should try avoiding using these functions

function LinearAlgebra.kron(a::AbstractRank1Tensor)
  kron(reverse(get_factors(a))...)
end

function LinearAlgebra.kron(a::AbstractRankTensor)
  sum([kron(get_decomposition(a,k)) for k in eachindex(a)])
end
