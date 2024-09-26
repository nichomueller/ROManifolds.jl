abstract type AbstractRankTensor{D,K} end

dimension(a::AbstractRankTensor{D,K}) where {D,K} = D
LinearAlgebra.rank(a::AbstractRankTensor{D,K}) where {D,K} = K
get_decomposition(a::AbstractRankTensor) = @abstractmethod

struct Rank1Tensor{D,A<:AbstractArray} <: AbstractRankTensor{D,1}
  factors::Vector{A}
  function Rank1Tensor(factors::Vector{A}) where A
    D = length(factors)
    new{D,A}(factors)
  end
end

get_factors(a::Rank1Tensor) = a.factors
get_decomposition(a::Rank1Tensor) = get_factors(a)
Base.size(a::Rank1Tensor) = (dimension(a),)
Base.getindex(a::Rank1Tensor,d::Integer) = get_factors(a)[d]

struct GenericRankTensor{D,K,A<:AbstractArray} <: AbstractRankTensor{D,K}
  decompositions::Vector{Rank1Tensor{D,A}}
  function GenericRankTensor(decompositions::Vector{Rank1Tensor{D,A}}) where {D,A}
    K = length(decompositions)
    new{D,K,A}(decompositions)
  end
end

get_decomposition(a::GenericRankTensor) = ntuple(k -> getindex(a,k),Val{rank(a)}())
get_decomposition(a::GenericRankTensor,k::Integer) = a.decompositions[k]
get_factor(a::GenericRankTensor,d::Integer,k::Integer) = get_factor(get_decomposition(a,k),d)
Base.size(a::GenericRankTensor) = (rank(a),)
Base.getindex(a::GenericRankTensor,k::Integer) = get_decomposition(a,k)

function _sort!(a::AbstractVector{<:AbstractVector},i::Tuple{<:TProductIndexMap})
  irow, = i
  irow1d = IndexMaps.get_univariate_indices(irow)
  for (i,(ai,irowi)) in enumerate(zip(a,irow1d))
    a[i] = ai[vec(irowi)]
  end
end

function _sort!(a::AbstractVector{<:AbstractMatrix},i::Tuple{<:TProductIndexMap,<:TProductIndexMap})
  irow,icol = i
  irow1d = IndexMaps.get_univariate_indices(irow)
  icol1d = IndexMaps.get_univariate_indices(icol)
  for (i,(ai,irowi,icoli)) in enumerate(zip(a,irow1d,icol1d))
    a[i] = ai[vec(irowi),vec(icoli)]
  end
end

function tproduct_array(arrays_1d::Vector{<:AbstractArray},index_map)
  _sort!(arrays_1d,index_map)
  Rank1Tensor(arrays_1d)
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
    return Rank1Tensor(di)
  end
  return d
end

function _find_decompositions(summation,arrays_1d,gradients_1d)
  @check length(arrays_1d) == length(gradients_1d)
  inds = LinearIndices(arrays_1d)
  d = map(inds) do i
    di = copy(arrays_1d)
    di[i] += gradients_1d[i]
    return Rank1Tensor(di)
  end
  return d
end

# MultiField interface
struct BlockRankTensor{A<:AbstractRankTensor,N} <: AbstractArray{A,N}
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
  BlockRankTensor(arrays)
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
  BlockRankTensor(arrays)
end

Base.size(a::BlockRankTensor) = size(a.array)

function Base.getindex(
  a::BlockRankTensor{A,N},
  i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  getindex(a.array,i...)
end

function Base.setindex!(
  a::BlockRankTensor{A,N},
  v,i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  setindex!(a.array,v,i...)
end

function Base.getindex(a::BlockRankTensor{A,N},i::Block{N}) where {A,N}
  getindex(a.array,i.n...)
end

function primal_dual_blocks(a::BlockRankTensor)
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

function tpmul(a::AbstractRankTensor{2,1},b::AbstractMatrix)
  return a[1]*b*a[2]'
end

function tpmul(a::AbstractRankTensor{3,1},b::AbstractArray{T,3} where T)
  hcat([vec(a[1]*bi*a[2]') for bi in eachslice(b,dims=3)]...)*a[3]'
end

function tpmul(a::AbstractRankTensor{D,K},b::AbstractArray) where {D,K}
  c = tpmul(get_decomposition(a,1),b)
  for k = 2:K
    c += tpmul(get_decomposition(a,k),b)
  end
  return c
end

Utils.induced_norm(a::AbstractArray,X::AbstractRankTensor) = sqrt(dot(vec(a),tpmul(X,a)))

# to global array - should try avoiding using these functions

function LinearAlgebra.kron(a::AbstractRankTensor{D,1}) where D
  kron(reverse(get_factors(a))...)
end

function LinearAlgebra.kron(a::AbstractRankTensor{D,K}) where {D,K}
  sum([kron(get_decomposition(a,k)) for k in 1:K])
end
