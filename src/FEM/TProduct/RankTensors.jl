abstract type AbstractRankTensor{D,K} end

dimension(a::AbstractRankTensor{D,K}) where {D,K} = D
LinearAlgebra.rank(a::AbstractRankTensor{D,K}) where {D,K} = K
get_decomposition(a::AbstractRankTensor) = ntuple(k -> get_decomposition(a,k),Val{rank(a)}())
get_decomposition(a::AbstractRankTensor,k::Integer) = @abstractmethod

struct Rank1Tensor{D,A<:AbstractArray} <: AbstractRankTensor{D,1}
  factors::Vector{A}
  function Rank1Tensor(factors::Vector{A}) where A
    D = length(factors)
    new{D,A}(factors)
  end
end

get_factors(a::Rank1Tensor) = a.factors
get_decomposition(a::Rank1Tensor,k::Integer) = k == 1 ? a : error("Exceeded rank 1 with rank $k")
Base.size(a::Rank1Tensor) = (dimension(a),)
Base.getindex(a::Rank1Tensor,d::Integer) = get_factors(a)[d]

struct GenericRankTensor{D,K,A<:AbstractArray} <: AbstractRankTensor{D,K}
  decompositions::Vector{Rank1Tensor{D,A}}
  function GenericRankTensor(decompositions::Vector{Rank1Tensor{D,A}}) where {D,A}
    K = length(decompositions)
    new{D,K,A}(decompositions)
  end
end

get_decomposition(a::GenericRankTensor,k::Integer) = a.decompositions[k]
get_factor(a::GenericRankTensor,d::Integer,k::Integer) = get_factor(get_decomposition(a,k),d)
Base.size(a::GenericRankTensor) = (rank(a),)
Base.getindex(a::GenericRankTensor,k::Integer) = get_decomposition(a,k)

function _sort!(a::AbstractVector{<:AbstractVector},i::Tuple{<:TProductDofMap})
  irow, = i
  irow1d = DofMaps.get_univariate_dof_map(irow)
  for (i,(ai,irowi)) in enumerate(zip(a,irow1d))
    a[i] = ai[vec(irowi)]
  end
end

function _sort!(a::AbstractVector{<:AbstractMatrix},i::Tuple{<:TProductDofMap,<:TProductDofMap})
  irow,icol = i
  irow1d = DofMaps.get_univariate_dof_map(irow)
  icol1d = DofMaps.get_univariate_dof_map(icol)
  for (i,(ai,irowi,icoli)) in enumerate(zip(a,irow1d,icol1d))
    a[i] = ai[vec(irowi),vec(icoli)]
  end
end

function tproduct_array(arrays_1d::Vector{<:AbstractArray},dof_map)
  _sort!(arrays_1d,dof_map)
  Rank1Tensor(arrays_1d)
end

function tproduct_array(
  ::Nothing,
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  dof_map,
  summation=nothing)

  tproduct_array(arrays_1d,dof_map)
end

function tproduct_array(
  ::typeof(gradient),
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  dof_map,
  summation=nothing)

  _sort!(arrays_1d,dof_map)
  _sort!(gradients_1d,dof_map)
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

function tproduct_array(arrays_1d::Vector{<:BlockArray},dof_map)
  s_blocks = blocksize(first(arrays_1d))
  arrays = map(CartesianIndices(s_blocks)) do i
    iblock = Block(Tuple(i))
    arrays_1d_i = getindex.(arrays_1d,iblock)
    dof_map_i = getindex.(dof_map,Tuple(i))
    tproduct_array(arrays_1d_i,dof_map_i)
  end
  BlockRankTensor(arrays)
end

function tproduct_array(op::ArrayBlock,arrays_1d::Vector{<:BlockArray},gradients_1d::Vector{<:BlockArray},dof_map,s::ArrayBlock)
  s_blocks = blocksize(first(arrays_1d))
  arrays = map(CartesianIndices(s_blocks)) do i
    iblock = Block(Tuple(i))
    arrays_1d_i = getindex.(arrays_1d,iblock)
    gradients_1d_i = getindex.(gradients_1d,iblock)
    dof_map_i = getindex.(dof_map,Tuple(i))
    op_i = op[Tuple(i)...]
    s_i = s[Tuple(i)...]
    tproduct_array(op_i,arrays_1d_i,gradients_1d_i,dof_map_i,s_i)
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

# linear algebra

function tpmul(a::Rank1Tensor{2},b::AbstractMatrix)
  return a[1]*b*a[2]'
end

function tpmul(a::Rank1Tensor{3},b::AbstractArray{T,3} where T)
  hcat([vec(a[1]*bi*a[2]') for bi in eachslice(b,dims=3)]...)*a[3]'
end

function tpmul(a::AbstractRankTensor{D,K},b::AbstractArray) where {D,K}
  c = tpmul(get_decomposition(a,1),b)
  for k = 2:K
    c += tpmul(get_decomposition(a,k),b)
  end
  return c
end

function Utils.induced_norm(a::AbstractArray{T,D},X::AbstractRankTensor{D}) where {T,D}
  sqrt(dot(vec(a),vec(tpmul(X,a))))
end

function Utils.induced_norm(a::AbstractArray{T,D′},X::AbstractRankTensor{D}) where {T,D,D′}
  D ≥ D′ && @notimplemented
  sqrt(sum(induced_norm(ai,X)^2 for ai in eachslice(a,dims=D′)))
end

# to global array - should try avoiding using these functions

function LinearAlgebra.kron(a::AbstractRankTensor{D,1}) where D
  kron(reverse(get_factors(a))...)
end

function LinearAlgebra.kron(a::AbstractRankTensor{D,K}) where {D,K}
  sum([kron(get_decomposition(a,k)) for k in 1:K])
end
