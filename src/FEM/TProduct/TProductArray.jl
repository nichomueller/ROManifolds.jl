"""
    abstract type AbstractTProductArray end

Type storing information related to FE arrays in a 1-D setting. The FE array is
defined as their permuted (see [`get_tp_dof_index_map`](@ref)) tensor product.

Subtypes:
- [`TProductArray`](@ref)
- [`TProductGradientArray`](@ref)
- [`TProductDivergenceArray`](@ref)
- [`BlockTProductArray`](@ref)

"""
abstract type AbstractTProductArray end

get_arrays(a::AbstractTProductArray) = @abstractmethod
IndexMaps.get_index_map(a::AbstractTProductArray) = @abstractmethod
tp_length(a::AbstractTProductArray) = length(get_arrays(a))

function tp_getindex(a::AbstractTProductArray,d::Integer)
  D = tp_length(a)
  @check d ≤ D
  tp_getindex(a,Val{d}(),Val{D}())
end

"""
    TProductArray{T,N,A,I} <: AbstractTProductArray

Represents a mass matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    M₁₂₃... = U+1D4D8(M₁ ⊗ M₂ ⊗ M₃ ⊗ ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
U+1D4D8(⋅) is the index map, and M₁₂₃... is the D-dimensional mass matrix

"""
struct TProductArray{T,N,A<:AbstractArray{T,N},I} <: AbstractTProductArray
  arrays_1d::Vector{A}
  index_map::I
end

function tproduct_array(arrays_1d::Vector{<:AbstractArray},index_map)
  TProductArray(arrays_1d,index_map)
end

get_arrays(a::TProductArray) = a.arrays_1d
IndexMaps.get_index_map(a::TProductArray) = a.index_map

tp_getindex(a::TProductArray,::Val{d},::Val{D}) where {d,D} = get_arrays(a)[d]

"""
    TProductGradientArray{T,N,A,I,B} <: AbstractTProductArray

Represents a stiffness matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    A₁₂₃... = U+1D4D8(A₁ ⊗ M₂ ⊗ M₃ ⊗ ... + M₁ ⊗ A₂ ⊗ M₃ ⊗ ... + ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
A₁, A₂, A₃, ... represent the 1-D stiffness matrices on their respective axes,
U+1D4D8(⋅) is the index map, and A₁₂₃... is the D-dimensional stiffness matrix.
The field `summation` (`nothing` by default) can represent a sum operation, in
which case the mass matrix is added to the stiffness

"""
struct TProductGradientArray{T,N,A<:AbstractArray{T,N},I,B} <: AbstractTProductArray
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
  index_map::I
  summation::B
end

function tproduct_array(
  ::typeof(gradient),
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  index_map,
  summation=nothing)

  if all(iszero.(gradients_1d)) || all(isempty.(gradients_1d))
    TProductArray(arrays_1d,index_map)
  else
    TProductGradientArray(arrays_1d,gradients_1d,index_map,summation)
  end
end

get_arrays(a::TProductGradientArray) = a.arrays_1d
get_gradients(a::TProductGradientArray) = a.gradients_1d

IndexMaps.get_index_map(a::TProductGradientArray) = a.index_map

tp_getindex(a::TProductGradientArray,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_getindex(a::TProductGradientArray,::Val{1},::Val{1}) = [get_arrays(a)[1],get_gradients(a)[1]]
tp_getindex(a::TProductGradientArray,::Val{1},::Val{2}) = [get_arrays(a)[1],get_gradients(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientArray,::Val{2},::Val{2}) = [get_arrays(a)[2],get_arrays(a)[2],get_gradients(a)[2]]
tp_getindex(a::TProductGradientArray,::Val{1},::Val{3}) = [get_arrays(a)[1],get_gradients(a)[1],get_arrays(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientArray,::Val{2},::Val{3}) = [get_arrays(a)[2],get_arrays(a)[2],get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductGradientArray,::Val{3},::Val{3}) = [get_arrays(a)[3],get_arrays(a)[3],get_arrays(a)[3],get_gradients(a)[3]]

const TProductGradientOnly{T,N,A,I} = TProductGradientArray{T,N,A,I,Nothing}

tp_getindex(a::TProductGradientOnly,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_getindex(a::TProductGradientOnly,::Val{1},::Val{1}) = [get_gradients(a)[1]]
tp_getindex(a::TProductGradientOnly,::Val{1},::Val{2}) = [get_gradients(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientOnly,::Val{2},::Val{2}) = [get_arrays(a)[2],get_gradients(a)[2]]
tp_getindex(a::TProductGradientOnly,::Val{1},::Val{3}) = [get_gradients(a)[1],get_arrays(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductGradientOnly,::Val{2},::Val{3}) = [get_arrays(a)[2],get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductGradientOnly,::Val{3},::Val{3}) = [get_arrays(a)[3],get_arrays(a)[3],get_gradients(a)[3]]

"""
    TProductDivergenceArray{T,N,A,I} <: AbstractTProductArray

Represents a pressure-velocity matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    B₁₂₃... = U+1D4D8(B₁ ⊗ M₂ ⊗ M₃ ⊗ ... + M₁ ⊗ B₂ ⊗ M₃ ⊗ ... + ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
B₁, B₂, B₃, ... represent the 1-D pressure-velocity matrices on their respective axes,
U+1D4D8(⋅) is the index map, and B₁₂₃... is the D-dimensional pressure-velocity matrix

"""
struct TProductDivergenceArray{T,N,A<:AbstractArray{T,N},I} <: AbstractTProductArray
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
  index_map::I
end

function tproduct_array(
  ::typeof(divergence),
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  index_map,
  summation=nothing)

  @notimplementedif (all(iszero.(gradients_1d) || isempty.(gradients_1d)))
  @notimplementedif !isnothing(summation)
  TProductDivergenceArray(arrays_1d,gradients_1d,index_map)
end

get_arrays(a::TProductDivergenceArray) = a.arrays_1d
get_gradients(a::TProductDivergenceArray) = a.gradients_1d

IndexMaps.get_index_map(a::TProductDivergenceArray) = a.index_map

tp_getindex(a::TProductDivergenceArray,::Val{d},::Val{D}) where {d,D} = @notimplemented
tp_getindex(a::TProductDivergenceArray,::Val{1},::Val{1}) = [get_gradients(a)[1]]
tp_getindex(a::TProductDivergenceArray,::Val{1},::Val{2}) = [get_arrays(a)[1],get_gradients(a)[1]]
tp_getindex(a::TProductDivergenceArray,::Val{2},::Val{2}) = [get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductDivergenceArray,::Val{1},::Val{3}) = [get_gradients(a)[1],get_arrays(a)[1],get_arrays(a)[1]]
tp_getindex(a::TProductDivergenceArray,::Val{2},::Val{3}) = [get_arrays(a)[2],get_gradients(a)[2],get_arrays(a)[2]]
tp_getindex(a::TProductDivergenceArray,::Val{3},::Val{3}) = [get_arrays(a)[3],get_arrays(a)[3],get_gradients(a)[3]]

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

for T in (:(typeof(gradient)),:(typeof(divergence)))
  @eval begin
    function tproduct_array(op::$T,arrays_1d::Vector{<:BlockArray},gradients_1d::Vector{<:BlockArray},index_map,s::ArrayBlock)
      s_blocks = blocksize(first(arrays_1d))
      arrays = map(CartesianIndices(s_blocks)) do i
        iblock = Block(Tuple(i))
        arrays_1d_i = getindex.(arrays_1d,iblock)
        gradients_1d_i = getindex.(gradients_1d,iblock)
        index_map_i = getindex.(index_map,Tuple(i))
        s_i = s[Tuple(i)...]
        tproduct_array(op,arrays_1d_i,gradients_1d_i,index_map_i,s_i)
      end
      BlockTProductArray(arrays)
    end
  end
end

Base.size(a::BlockTProductArray) = size(a.array)

Base.@propagate_inbounds function Base.getindex(
  a::BlockTProductArray{A,N},
  i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  getindex(a.array,i...)
end

Base.@propagate_inbounds function Base.setindex!(
  a::BlockTProductArray{A,N},
  v,i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  setindex!(a.array,v,i...)
end

Base.@propagate_inbounds function Base.getindex(a::BlockTProductArray{A,N},i::Block{N}) where {A,N}
  getindex(a.array,i.n...)
end

# to global array

function _kron(a::AbstractArray...)
  kron(reverse(a)...)
end

function LinearAlgebra.kron(a::TProductArray{T,2}) where T
  rows,cols = get_index_map(a)
  kp = _kron(get_arrays(a)...)
  kp[vec(rows),vec(cols)]
end

function LinearAlgebra.kron(a::TProductGradientArray{T,2}) where T
  rows,cols = get_index_map(a)
  _kron(Val{tp_length(a)}(),a,rows,cols)
end

function LinearAlgebra.kron(a::TProductDivergenceArray{T,2}) where T
  rows,cols = get_index_map(a)
  _kron(Val{tp_length(a)}(),a,rows,cols)
end

function _kron(::Val{d},a::TProductGradientArray,rows::TProductIndexMap,cols::TProductIndexMap) where d
  _kron(Val{d}(),a,rows.indices,cols.indices)
end
function _kron(::Val{d},a::TProductGradientArray,rows::AbstractIndexMap,cols::AbstractIndexMap) where d
  _kron(Val{d}(),a)[vec(rows),vec(cols)]
end
function _kron(::Val{d},a::TProductGradientArray,rows::AbstractMultiValueIndexMap,cols::AbstractMultiValueIndexMap) where d
  kp = _kron(Val{d}(),a)[vec(rows),vec(cols)]
  @check num_components(rows) == num_components(cols)
  ncomps = num_components(rows)
  row_blocks = cumsum(map(i -> length(get_component(rows,i)),1:ncomps))
  col_blocks = cumsum(map(i -> length(get_component(rows,i)),1:ncomps))
  pushfirst!(row_blocks,0)
  pushfirst!(col_blocks,0)
  for icomp in 1:ncomps, jcomp in icomp+1:ncomps
    irows = row_blocks[icomp]+1:row_blocks[icomp+1]
    jcols = col_blocks[jcomp]+1:col_blocks[jcomp+1]
    kp[irows,jcols] .= zero(eltype(kp))
    kp[jcols,irows] .= zero(eltype(kp))
  end
  kp
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
  (
    _kron(get_arrays(a)[1],get_gradients(a)[2]) +
    _kron(get_gradients(a)[1],get_arrays(a)[2])
  )
end
function _kron(::Val{3},a::TProductGradientOnly)
  (
    _kron(get_gradients(a)[1],get_arrays(a)[2],get_arrays(a)[3]) +
    _kron(get_arrays(a)[1],get_gradients(a)[2],get_arrays(a)[3]) +
    _kron(get_arrays(a)[1],get_arrays(a)[2],get_gradients(a)[3])
  )
end

function _kron(::Val{d},a::TProductDivergenceArray,rows::AbstractArray,cols::AbstractArray) where d
  @notimplemented
end
function _kron(::Val{d},a::TProductDivergenceArray,rows::TProductIndexMap,cols::TProductIndexMap) where d
  _kron(Val{d}(),a,rows.indices,cols.indices)
end
function _kron(::Val{d},a::TProductDivergenceArray,rows::AbstractMultiValueIndexMap,cols::FixedDofsIndexMap) where d
  _kron(Val{d}(),a,rows,remove_fixed_dof(cols))
end
function _kron(::Val{d},a::TProductDivergenceArray,rows::FixedDofsIndexMap,cols::AbstractMultiValueIndexMap) where d
  _kron(Val{d}(),a,remove_fixed_dof(rows),cols)
end
function _kron(::Val{d},a::TProductDivergenceArray,rows::FixedDofsIndexMap,cols::FixedDofsIndexMap) where d
  _kron(Val{d}(),a,remove_fixed_dof(rows),remove_fixed_dof(cols))
end
function _kron(::Val{d},a::TProductDivergenceArray,rows::AbstractMultiValueIndexMap,cols::AbstractArray) where d
  vcat(
    _kron(get_gradients(a)[1],get_arrays(a)[2]),
    _kron(get_arrays(a)[1],get_gradients(a)[2])
  )[vec(rows),cols]
end
function _kron(::Val{d},a::TProductDivergenceArray,cols::AbstractArray,rows::AbstractMultiValueIndexMap) where d
  hcat(
    _kron(get_gradients(a)[1],get_arrays(a)[2]),
    _kron(get_arrays(a)[1],get_gradients(a)[2])
  )[vec(rows),cols]
end
