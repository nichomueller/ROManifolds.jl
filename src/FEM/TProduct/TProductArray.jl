"""
    abstract type AbstractTProductArray end

Type storing information related to FE arrays in a 1-D setting. The FE array is
defined as their permuted (see [`get_tp_dof_index_map`](@ref)) tensor product.

Subtypes:
- [`TProductArray`](@ref)
- [`TProductGradientArray`](@ref)
- [`TProductPDerivativeArray`](@ref)
- [`BlockTProductArray`](@ref)

"""
abstract type AbstractTProductArray end

get_arrays(a::AbstractTProductArray) = @abstractmethod
tp_length(a::AbstractTProductArray) = length(get_arrays(a))

function tp_getindex(a::AbstractTProductArray,d::Integer)
  D = tp_length(a)
  @check d ≤ D
  tp_getindex(a,Val{d}(),Val{D}())
end

function tp_decomposition(a::AbstractTProductArray)
  D = tp_length(a)
  T = eltype(get_arrays(a))
  ka = T[]
  for d in 1:tp_length(a)+1
    try kad = tp_decomposition(a,Val{d}(),Val{D}())
      push!(ka,kad)
    catch
      break
    end
  end
  return ka
end

function tp_sort!(a::AbstractVector{<:AbstractVector},i::NTuple{1,<:TProductIndexMap})
  irow, = i
  irow1d = IndexMaps.get_univariate_indices(irow)
  for (i,(ai,irowi)) in enumerate(zip(a,irow1d))
    a[i] = ai[vec(irowi)]
  end
end

function tp_sort!(a::AbstractVector{<:AbstractMatrix},i::NTuple{2,<:TProductIndexMap})
  irow,icol = i
  irow1d = IndexMaps.get_univariate_indices(irow)
  icol1d = IndexMaps.get_univariate_indices(icol)
  for (i,(ai,irowi,icoli)) in enumerate(zip(a,irow1d,icol1d))
    a[i] = ai[vec(irowi),vec(icoli)]
  end
end

"""
    TProductArray{T,N,A} <: AbstractTProductArray

Represents a mass matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    M₁₂₃... = U+1D4D8(M₁ ⊗ M₂ ⊗ M₃ ⊗ ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
U+1D4D8(⋅) is the index map, and M₁₂₃... is the D-dimensional mass matrix

"""
struct TProductArray{T,N,A<:AbstractArray{T,N}} <: AbstractTProductArray
  arrays_1d::Vector{A}
end

function tproduct_array(arrays_1d::Vector{<:AbstractArray},index_map)
  tp_sort!(arrays_1d,index_map)
  TProductArray(arrays_1d)
end

get_arrays(a::TProductArray) = a.arrays_1d

tp_getindex(a::TProductArray,::Val{d},::Val{D}) where {d,D} = [get_arrays(a)[d]]
tp_decomposition(a::TProductArray,::Val{d},::Val{D}) where {d,D} = get_arrays(a)

"""
    TProductGradientArray{T,N,A,B} <: AbstractTProductArray

Represents a stiffness matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    A₁₂₃... = U+1D4D8(A₁ ⊗ M₂ ⊗ M₃ ⊗ ... + M₁ ⊗ A₂ ⊗ M₃ ⊗ ... + ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
A₁, A₂, A₃, ... represent the 1-D stiffness matrices on their respective axes,
U+1D4D8(⋅) is the index map, and A₁₂₃... is the D-dimensional stiffness matrix.
The field `summation` (`nothing` by default) can represent a sum operation, in
which case the mass matrix is added to the stiffness

"""
struct TProductGradientArray{T,N,A<:AbstractArray{T,N},B} <: AbstractTProductArray
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
  summation::B
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

  tp_sort!(arrays_1d,index_map)
  tp_sort!(gradients_1d,index_map)
  TProductGradientArray(arrays_1d,gradients_1d,summation)
end

get_arrays(a::TProductGradientArray) = a.arrays_1d
get_gradients(a::TProductGradientArray) = a.gradients_1d

tp_getindex(a::TProductGradientArray,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_getindex(a::TProductGradientArray,::Val{1},::Val{1}) = [get_arrays(a)[1],get_gradients(a)[1]]
tp_getindex(a::TProductGradientArray,::Val{1},::Val{2}) = [get_arrays(a)[1],get_gradients(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientArray,::Val{2},::Val{2}) = [get_arrays(a)[2],get_arrays(a)[2],get_gradients(a)[2]]
tp_getindex(a::TProductGradientArray,::Val{1},::Val{3}) = [get_arrays(a)[1],get_gradients(a)[1],get_arrays(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientArray,::Val{2},::Val{3}) = [get_arrays(a)[2],get_arrays(a)[2],get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductGradientArray,::Val{3},::Val{3}) = [get_arrays(a)[3],get_arrays(a)[3],get_arrays(a)[3],get_gradients(a)[3]]

tp_decomposition(a::TProductGradientArray,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_decomposition(a::TProductGradientArray,::Val{1},::Val{2}) = get_arrays(a)
tp_decomposition(a::TProductGradientArray,::Val{2},::Val{2}) = [get_gradients(a)[1],get_arrays(a)[2]]
tp_decomposition(a::TProductGradientArray,::Val{3},::Val{2}) = [get_arrays(a)[1],get_gradients(a)[2]]
tp_decomposition(a::TProductGradientArray,::Val{1},::Val{3}) = get_arrays(a)
tp_decomposition(a::TProductGradientArray,::Val{2},::Val{3}) = [get_gradients(a)[1],get_arrays(a)[2],get_arrays(a)[3]]
tp_decomposition(a::TProductGradientArray,::Val{3},::Val{3}) = [get_arrays(a)[1],get_gradients(a)[2],get_arrays(a)[3]]
tp_decomposition(a::TProductGradientArray,::Val{4},::Val{3}) = [get_arrays(a)[1],get_arrays(a)[2],get_gradients(a)[3]]

const TProductGradientOnly{T,N,A} = TProductGradientArray{T,N,A,Nothing}

tp_getindex(a::TProductGradientOnly,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_getindex(a::TProductGradientOnly,::Val{1},::Val{2}) = [get_gradients(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientOnly,::Val{2},::Val{2}) = [get_arrays(a)[2],get_gradients(a)[2]]
tp_getindex(a::TProductGradientOnly,::Val{1},::Val{3}) = [get_gradients(a)[1],get_arrays(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientOnly,::Val{2},::Val{3}) = [get_arrays(a)[2],get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductGradientOnly,::Val{3},::Val{3}) = [get_arrays(a)[3],get_arrays(a)[3],get_gradients(a)[3]]

tp_decomposition(a::TProductGradientOnly,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_decomposition(a::TProductGradientOnly,::Val{1},::Val{2}) = [get_gradients(a)[1],get_arrays(a)[2]]
tp_decomposition(a::TProductGradientOnly,::Val{2},::Val{2}) = [get_arrays(a)[1],get_gradients(a)[2]]
tp_decomposition(a::TProductGradientOnly,::Val{1},::Val{3}) = [get_gradients(a)[1],get_arrays(a)[2],get_arrays(a)[3]]
tp_decomposition(a::TProductGradientOnly,::Val{2},::Val{3}) = [get_arrays(a)[1],get_gradients(a)[2],get_arrays(a)[3]]
tp_decomposition(a::TProductGradientOnly,::Val{3},::Val{3}) = [get_arrays(a)[1],get_arrays(a)[2],get_gradients(a)[3]]

"""
    TProductPDerivativeArray{O,T,N,A} <: AbstractTProductArray

Represents a pressure-velocity matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    B₁₂₃... = U+1D4D8(B₁ ⊗ M₂ ⊗ M₃ ⊗ ... + M₁ ⊗ B₂ ⊗ M₃ ⊗ ... + ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
B₁, B₂, B₃, ... represent the 1-D pressure-velocity matrices on their respective axes,
U+1D4D8(⋅) is the index map, and B₁₂₃... is the D-dimensional pressure-velocity matrix

"""
struct TProductPDerivativeArray{O,T,N,A<:AbstractArray{T,N}} <: AbstractTProductArray
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
  function TProductPDerivativeArray{O}(
    arrays_1d::Vector{A},
    gradients_1d::Vector{A}
    ) where {O,T,N,A<:AbstractArray{T,N}}

    @check O ∈ (1,2,3)
    new{O,T,N,A}(arrays_1d,gradients_1d)
  end
end

const TProductPDerivativeArray1{T,N,A<:AbstractArray{T,N}} = TProductPDerivativeArray{1,T,N,A}
const TProductPDerivativeArray2{T,N,A<:AbstractArray{T,N}} = TProductPDerivativeArray{2,T,N,A}
const TProductPDerivativeArray3{T,N,A<:AbstractArray{T,N}} = TProductPDerivativeArray{3,T,N,A}

get_arrays(a::TProductPDerivativeArray) = a.arrays_1d
get_gradients(a::TProductPDerivativeArray) = a.gradients_1d

tp_getindex(a::TProductPDerivativeArray,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_getindex(a::TProductPDerivativeArray,::Val{1},::Val{2}) = [get_arrays(a)[1],get_gradients(a)[1]]
tp_getindex(a::TProductPDerivativeArray,::Val{2},::Val{2}) = [get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductPDerivativeArray,::Val{1},::Val{3}) = [get_gradients(a)[1],get_arrays(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductPDerivativeArray,::Val{2},::Val{3}) = [get_arrays(a)[2],get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductPDerivativeArray,::Val{3},::Val{3}) = [get_arrays(a)[3],get_arrays(a)[3],get_gradients(a)[3]]

tp_decomposition(a::TProductPDerivativeArray) = tp_decomposition(a,Val{tp_length(a)}())
tp_decomposition(a::TProductPDerivativeArray,::Val{D}) where D = @notimplemented
tp_decomposition(a::TProductPDerivativeArray1,::Val{2}) = [get_gradients(a)[1],get_arrays(a)[2]]
tp_decomposition(a::TProductPDerivativeArray2,::Val{2}) = [get_arrays(a)[1],get_gradients(a)[2]]
tp_decomposition(a::TProductPDerivativeArray1,::Val{3}) = [get_gradients(a)[1],get_arrays(a)[2],get_arrays(a)[3]]
tp_decomposition(a::TProductPDerivativeArray2,::Val{3}) = [get_arrays(a)[1],get_gradients(a)[2],get_arrays(a)[3]]
tp_decomposition(a::TProductPDerivativeArray3,::Val{3}) = [get_arrays(a)[1],get_arrays(a)[2],get_gradients(a)[3]]

# MultiField interface
struct BlockTProductArray{A<:AbstractTProductArray,N} <: AbstractArray{A,N}
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
  BlockTProductArray(arrays)
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
  BlockTProductArray(arrays)
end

Base.size(a::BlockTProductArray) = size(a.array)

function Base.getindex(
  a::BlockTProductArray{A,N},
  i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  getindex(a.array,i...)
end

function Base.setindex!(
  a::BlockTProductArray{A,N},
  v,i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  setindex!(a.array,v,i...)
end

function Base.getindex(a::BlockTProductArray{A,N},i::Block{N}) where {A,N}
  getindex(a.array,i.n...)
end

function primal_dual_blocks(a::BlockTProductArray)
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

# to global array

function _kron(a::AbstractArray...)
  kron(reverse(a)...)
end

function LinearAlgebra.kron(a::TProductArray{T,2}) where T
  _kron(get_arrays(a)...)
end

function LinearAlgebra.kron(a::TProductGradientArray{T,2}) where T
  _kron(Val{tp_length(a)}(),a)
end

function _kron(::Val{1},a::TProductGradientArray)
  a.summation(get_arrays(a)[1],get_gradients(a)[1])
end

function _kron(::Val{2},a::TProductGradientArray)
  (
    a.summation(
      _kron(get_arrays(a)...),
      _kron(get_arrays(a)[1],get_gradients(a)[2]) +
      _kron(get_gradients(a)[1],get_arrays(a)[2])
    )
  )
end

function _kron(::Val{3},a::TProductGradientArray)
  (
    a.summation(
      _kron(get_arrays(a)...),
      _kron(get_gradients(a)[1],get_arrays(a)[2],get_arrays(a)[3]) +
      _kron(get_arrays(a)[1],get_gradients(a)[2],get_arrays(a)[3]) +
      _kron(get_arrays(a)[1],get_arrays(a)[2],get_gradients(a)[3])
    )
  )
end

function _kron(::Val{1},a::TProductGradientOnly)
  get_gradients(a)[1]
end

function _kron(::Val{2},a::TProductGradientOnly)
  hcat(
    _kron(get_arrays(a)[1],get_gradients(a)[2]),
    _kron(get_gradients(a)[1],get_arrays(a)[2])
  )
end

function _kron(::Val{3},a::TProductGradientOnly)
  hcat(
    _kron(get_gradients(a)[1],get_arrays(a)[2],get_arrays(a)[3]),
    _kron(get_arrays(a)[1],get_gradients(a)[2],get_arrays(a)[3]),
    _kron(get_arrays(a)[1],get_arrays(a)[2],get_gradients(a)[3])
  )
end
