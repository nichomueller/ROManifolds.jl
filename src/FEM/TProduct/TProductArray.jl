"""
    abstract type AbstractTProductArray{T,N} <: AbstractArray{T,N} end

Type storing information related to FE arrays in a 1-D setting, and the FE array
defined as their permuted tensor product. The index permutation is encoded in an
AbstractIndexMap, to be provided with factors and tensor product arrays.

Subtypes:
- [`TProductArray`](@ref)
- [`TProductGradientArray`](@ref)
- [`MultiValueTProductArray`](@ref)
- [`BlockTProductArray`](@ref)

"""
#TODO I'm only using this for matrices, not vectors. Can change to <: AbstractMatrix{T} ?
abstract type AbstractTProductArray{T,N} <: AbstractArray{T,N} end

Arrays.get_array(a::AbstractTProductArray) = @abstractmethod
get_index_map(a::AbstractTProductArray) = @abstractmethod
univariate_length(a::AbstractTProductArray) = @abstractmethod

Base.size(a::AbstractTProductArray) = size(get_array(a))

Base.@propagate_inbounds function _find_block_index(imap::Vector{<:TProductIndexMap},i::Int)
  slengths = cumsum(length.(imap))
  blocki = findfirst(i .≤ slengths)
  i0 = blocki>1 ? slengths[blocki-1] : 0
  imap[blocki][i-i0]
end

_map_getindex(args...) = @abstractmethod
Base.@propagate_inbounds _map_getindex(imap::AbstractIndexMap,i::Int) = (imap[i],)
Base.@propagate_inbounds _map_getindex(imap::AbstractIndexMap,jmap::AbstractIndexMap,i::Int,j::Int) = (imap[i],jmap[j])

Base.@propagate_inbounds function _map_getindex(
  imap::AbstractMultiValueIndexMap{D,Ti},
  jmap::AbstractMultiValueIndexMap{D,Ti},
  i::Int,
  j::Int,
  n::Int
  ) where {D,Ti}

  imapn = selectdim(imap,D,n)
  jmapn = selectdim(jmap,D,n)
  (imapn[i],jmapn[j])
end

Base.@propagate_inbounds function _map_getindex(
  imap::Vector{<:AbstractIndexMap},
  jmap::Vector{<:AbstractIndexMap},
  i::Int,j::Int)

  i′ = _find_block_index(imap,i)
  j′ = _find_block_index(jmap,j)
  (i′,j′)
end

Base.@propagate_inbounds function Base.getindex(a::AbstractTProductArray{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(a,i...)
  i′ = _map_getindex(get_index_map(a)...,i...)
  any(i′ .== 0) ? zero(eltype(a)) : getindex(get_array(a),i′...)
end

Base.@propagate_inbounds function Base.setindex!(a::AbstractTProductArray{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(a,i...)
  i′ = _map_getindex(get_index_map(a)...,i...)
  all(i′ .!= 0) && setindex!(get_array(a),v,i′...)
end

Base.fill!(a::AbstractTProductArray,v) = fill!(get_array(a),v)

function LinearAlgebra.mul!(
  c::AbstractTProductArray,
  a::AbstractTProductArray,
  b::AbstractTProductArray,
  α::Number,β::Number)

  mul!(get_array(c),get_array(a),get_array(b),α,β)
end

function LinearAlgebra.axpy!(α::Number,a::AbstractTProductArray,b::AbstractTProductArray)
  axpy!(α,get_array(a),get_array(b))
end

for factorization in (:LU,:Cholesky)
  @eval begin
    function LinearAlgebra.ldiv!(m::$factorization,b::AbstractTProductArray)
      ldiv!(m,get_array(b))
      return b
    end
  end
end

function LinearAlgebra.ldiv!(a::AbstractTProductArray,m::Factorization,b::AbstractTProductArray)
  ldiv!(get_array(a),m,get_array(b))
  return a
end

function LinearAlgebra.rmul!(a::AbstractTProductArray,b::Number)
  rmul!(get_array(a),b)
  return a
end

function LinearAlgebra.lu(a::AbstractTProductArray)
  lu(get_array(a))
end

function LinearAlgebra.lu!(a::AbstractTProductArray,b::AbstractTProductArray)
  lu!(get_array(a),get_array(b))
end

"""
    TProductArray{T,N,A,I} <: AbstractTProductArray{T,N}

Represents a mass matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    M₁₂₃... = U+1D4D8(M₁ ⊗ M₂ ⊗ M₃ ⊗ ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
U+1D4D8(⋅) is the index map, and M₁₂₃... is the D-dimensional mass matrix

"""
struct TProductArray{T,N,A<:AbstractArray{T,N},I} <: AbstractTProductArray{T,N}
  array::A
  arrays_1d::Vector{A}
  index_map::NTuple{N,I}
end

function tproduct_array(
  array::A,
  arrays_1d::Vector{A},
  index_map::NTuple{N,I}
  ) where {T,N,A<:AbstractArray{T,N},I<:TProductIndexMap}

  TProductArray(array,arrays_1d,index_map)
end

Arrays.get_array(a::TProductArray) = a.array
get_index_map(a::TProductArray) = a.index_map
univariate_length(a::TProductArray) = length(a.arrays_1d)

function tproduct_array(arrays_1d::Vector{A},index_map::NTuple) where A
  array::A = _kron(arrays_1d...)
  tproduct_array(array,arrays_1d,index_map)
end

function Base.similar(a::TProductArray{T,N},::Type{T′},dims::Dims{N}) where {T,T′,N}
  TProductArray(similar(a.array,T′,dims...),a.arrays_1d,a.index_map)
end

function Base.copyto!(a::TProductArray,b::TProductArray)
  @check size(a) == size(b)
  copyto!(a.array,a.array)
  map(copyto!,a.arrays_1d,b.arrays_1d)
  a
end

function IndexMaps.change_index_map(f,a::TProductArray)
  i = get_index_map(a)
  i′ = map(i->change_index_map(f,i),i)
  TProductArray(a.array,a.arrays_1d,i′)
end

const TProductSparseMatrix = TProductArray{T,2,A} where {T,A<:AbstractSparseMatrix}

SparseArrays.nnz(a::TProductSparseMatrix) = nnz(a.array)
SparseArrays.nzrange(a::TProductSparseMatrix,col::Integer) = nzrange(a.array,col)
SparseArrays.rowvals(a::TProductSparseMatrix) = rowvals(a.array)
SparseArrays.nonzeros(a::TProductSparseMatrix) = a.array

function _kron(A::AbstractArray...)
  kron(reverse(A)...)
end

function _kron(A::AbstractBlockArray...)
  map(zip(eachblock.(A)...)) do block
    _kron(block...)
  end |> mortar
end

function symbolic_kron(A::AbstractArray)
  A
end

function symbolic_kron(a::AbstractVector{T},b::AbstractVector{S}) where {T<:Number,S<:Number}
  c = Vector{promote_op(*,T,S)}(undef,length(a)*length(b))
  return c
end

function symbolic_kron(A::AbstractSparseMatrixCSC{T1,S1},B::AbstractSparseMatrixCSC{T2,S2}) where {T1,T2,S1,S2}
  mA,nA = size(A)
  mB,nB = size(B)
  mC,nC = mA*mB,nA*nB
  Tv = typeof(one(T1)*one(T2))
  Ti = promote_type(S1,S2)
  C = spzeros(Tv,Ti,mC,nC)
  sizehint!(C,nnz(A)*nnz(B))
  symbolic_kron!(C,B,A)
end

function symbolic_kron(A::AbstractBlockArray,B::AbstractBlockArray)
  map(zip(eachblock(A),eachblock(B))) do blockA,blockB
    symbolic_kron(blockA,blockB)
  end |> mortar
end

function symbolic_kron(A::AbstractArray,B::AbstractArray...)
  C,D... = B
  symbolic_kron(symbolic_kron(A,C),D...)
end

@inline function symbolic_kron!(C::SparseMatrixCSC,A::AbstractSparseMatrixCSC,B::AbstractSparseMatrixCSC)
  mA,nA = size(A)
  mB,nB = size(B)
  mC,nC = mA*mB,nA*nB

  msg = "target matrix needs to have size ($mC,$nC), but has size $(size(C))"
  @boundscheck size(C) == (mC,nC) || throw(DimensionMismatch(msg))

  rowvalC = rowvals(C)
  nzvalC = nonzeros(C)
  colptrC = getcolptr(C)

  nnzC = nnz(A)*nnz(B)
  resize!(nzvalC,nnzC)
  resize!(rowvalC,nnzC)

  col = 1
  @inbounds for j = 1:nA
    startA = getcolptr(A)[j]
    stopA = getcolptr(A)[j+1] - 1
    lA = stopA - startA + 1
    for i = 1:nB
      startB = getcolptr(B)[i]
      stopB = getcolptr(B)[i+1] - 1
      lB = stopB - startB + 1
      ptr_range = (1:lB) .+ (colptrC[col]-1)
      colptrC[col+1] = colptrC[col] + lA*lB
      col += 1
      for ptrA = startA : stopA
        ptrB = startB
        for ptr = ptr_range
          rowvalC[ptr] = (rowvals(A)[ptrA]-1)*mB + rowvals(B)[ptrB]
          ptrB += 1
        end
        ptr_range = ptr_range .+ lB
      end
    end
  end
  return C
end

function _numerical_kron!(A::AbstractArray,B::AbstractArray)
  copyto!(A,B)
  A
end

@inline function _numerical_kron!(c::Vector,a::AbstractVector{T},b::AbstractVector{S}) where {T<:Number,S<:Number}
  kron!(c,b,a)
  c
end

@inline function _numerical_kron!(C::SparseMatrixCSC,A::AbstractSparseMatrixCSC,B::AbstractSparseMatrixCSC)
  numerical_kron!(C,B,A)
  C
end

function _numerical_kron!(C::AbstractArray,A::AbstractArray,B::AbstractArray...)
  copyto!(C,_kron(A,B...))
  C
end

@inline function numerical_kron!(C::SparseMatrixCSC,A::AbstractSparseMatrixCSC,B::AbstractSparseMatrixCSC)
  nA = size(A,2)
  nB = size(B,2)

  nzvalC = nonzeros(C)
  colptrC = getcolptr(C)

  col = 1
  @inbounds for j = 1:nA
    startA = getcolptr(A)[j]
    stopA = getcolptr(A)[j+1] - 1
    for i = 1:nB
      startB = getcolptr(B)[i]
      stopB = getcolptr(B)[i+1] - 1
      lB = stopB - startB + 1
      ptr_range = (1:lB) .+ (colptrC[col]-1)
      col += 1
      for ptrA = startA : stopA
        ptrB = startB
        for ptr = ptr_range
          nzvalC[ptr] = nonzeros(A)[ptrA] * nonzeros(B)[ptrB]
          ptrB += 1
        end
        ptr_range = ptr_range .+ lB
      end
    end
  end
  return C
end

@inline function numerical_kron!(C::AbstractBlockArray,A::AbstractBlockArray,B::AbstractBlockArray)
  @inbounds for (blockC,blockA,blockB) in zip(eachblock(C),eachblock(A),eachblock(B))
    numerical_kron!(blockC,blockA,blockB)
  end
end

# for gradients

function kronecker_gradients(f,g,op=nothing)
  Df = length(f)
  Dg = length(g)
  @check Df == Dg
  _kronecker_gradients(f,g,op,Val(Df))
end

_kronecker_gradients(f,g,::Val{1}) = g[1]
_kronecker_gradients(f,g,::Val{2}) = _kron(g[1],f[2]) + _kron(f[1],g[2])
_kronecker_gradients(f,g,::Val{3}) = _kron(g[1],f[2],f[3]) + _kron(f[1],g[2],f[3]) + _kron(f[1],f[2],g[3])

_kronecker_gradients(f,g,::Nothing,::Val{d}) where d = _kronecker_gradients(f,g,Val(d))
_kronecker_gradients(f,g,op,::Val{d}) where d = op(_kron(f...),_kronecker_gradients(f,g,Val(d)))

function symbolic_kron(f,g)
  Df = length(f)
  Dg = length(g)
  @check Df == Dg
  _symbolic_kron(f,g,Val(Df))
end

_symbolic_kron(f,g,::Val{1}) = symbolic_kron(g[1])
_symbolic_kron(f,g,::Val{2}) = symbolic_kron(g[1],f[2])
_symbolic_kron(f,g,::Val{3}) = symbolic_kron(g[1],f[2],f[3])

@inline function _numerical_kron!(
  C::SparseMatrixCSC,
  vA::Vector{<:AbstractSparseMatrixCSC},
  vB::Vector{<:AbstractSparseMatrixCSC},
  args...)

  numerical_kron!(C,reverse(vA),reverse(vB),args...)
end

@inline function numerical_kron!(
  C::SparseMatrixCSC,
  vA::Vector{<:AbstractSparseMatrixCSC},
  vB::Vector{<:AbstractSparseMatrixCSC},
  op=nothing)

  _prod(f,g,::Val{1}) = g[1]
  _prod(f,g,::Val{2}) = g[1]*f[2] + f[1]*g[2]
  _prod(f,g,::Val{3}) = g[1]*f[2]*f[3] + f[1]*g[2]*f[3] + f[1]*f[2]*g[3]

  _prod(f,g,::Nothing,::Val{d}) where d = _prod(f,g,Val(d))
  _prod(f,g,op,::Val{d}) where d = op(prod(f),_prod(f,g,Val(d)))

  A = first(vA)
  B = first(vB)
  d = length(vA)

  nA = size(A,2)
  nB = size(B,2)

  nzvalvA = map(nonzeros,A)
  nzvalvB = map(nonzeros,B)
  cacheA = Vector{eltpye(nzvalC)}(undef,d)
  cacheB = Vector{eltpye(nzvalC)}(undef,d)

  nzvalC = nonzeros(C)
  colptrC = getcolptr(C)

  col = 1
  @inbounds for j = 1:nA
    startA = getcolptr(A)[j]
    stopA = getcolptr(A)[j+1] - 1
    for i = 1:nB
      startB = getcolptr(B)[i]
      stopB = getcolptr(B)[i+1] - 1
      lB = stopB - startB + 1
      ptr_range = (1:lB) .+ (colptrC[col]-1)
      col += 1
      for ptrA = startA : stopA
        ptrB = startB
        for ptr = ptr_range
          for di = 1:d
            cacheA[di] = nzvalvA[di][ptrA]
            cacheB[di] = nzvalvB[di][ptrB]
          end
          nzvalC[ptr] = _prod(cacheA,cacheB,op,Val(d))
          ptrB += 1
        end
        ptr_range = ptr_range .+ lB
      end
    end
  end
  return C
end

@inline function numerical_kron!(
  C::AbstractBlockArray,
  A::Vector{<:AbstractBlockArray},
  B::Vector{<:AbstractBlockArray})

  blocksA = map(eachblock)
  blockinds = CartesianIndices(axes.(blocklasts.(axes(mat)),1))
  for I in blockinds
    blockC = C[Block(Tuple(I))]
    blocksA = getindex.(A,Block(Tuple(I)))
    blocksB = getindex.(B,Block(Tuple(I)))
    numerical_kron!(blockC,blocksA,blocksB)
  end
end

"""
    TProductGradientArray{T,N,A,I} <: AbstractTProductArray{T,N}

Represents a stiffness matrix associated to a couple of tensor product FE spaces
[`TProductFESpace`](@ref). In fact:

    A₁₂₃... = U+1D4D8(M₁ ⊗ M₂ ⊗ M₃ ⊗ ... + A₁ ⊗ M₂ ⊗ M₃ ⊗ ... + M₁ ⊗ A₂ ⊗ M₃ ⊗ ... + ...),

where M₁, M₂, M₃, ... represent the 1-D mass matrices on their respective axes,
A₁, A₂, A₃, ... represent the 1-D stiffness matrices on their respective axes,
U+1D4D8(⋅) is the index map, and A₁₂₃... is the D-dimensional stiffness matrix

"""
struct TProductGradientArray{T,N,A<:AbstractArray{T,N},I} <: AbstractTProductArray{T,N}
  array::A
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
  index_map::NTuple{N,I}
end

function tproduct_array(
  array::A,
  arrays_1d::Vector{A},
  gradients_1d::Vector{A},
  index_map::NTuple{N,I}
  ) where {T,N,A<:AbstractArray{T,N},I<:TProductIndexMap}

  TProductGradientArray(array,arrays_1d,gradients_1d,index_map)
end

Arrays.get_array(a::TProductGradientArray) = a.array
get_index_map(a::TProductGradientArray) = a.index_map
univariate_length(a::TProductGradientArray) = length(a.arrays_1d)

function TProductGradientArray(arrays_1d::Vector{A},gradients_1d::Vector{A},index_map...) where A
  array::A = kronecker_gradients(arrays_1d,gradients_1d)
  TProductGradientArray(array,arrays_1d,gradients_1d,index_map...)
end

function Base.similar(a::TProductGradientArray{T,N},::Type{T′},dims::Dims{N}) where {T,T′,N}
  TProductGradientArray(similar(a.array,T′,dims...),a.arrays_1d,a.gradients_1d,a.index_map)
end

function Base.copyto!(a::TProductGradientArray,b::TProductGradientArray)
  @check size(a) == size(b)
  copyto!(a.array,a.array)
  map(copyto!,a.arrays_1d,b.arrays_1d)
  map(copyto!,a.gradients_1d,b.gradients_1d)
  a
end

function IndexMaps.change_index_map(f,a::TProductGradientArray)
  i = get_index_map(a)
  i′ = map(f,i)
  TProductGradientArray(a.array,a.arrays_1d,a.gradients_1d,i′)
end

const TProductGradientSparseMatrix{T,A<:AbstractSparseMatrix} = TProductGradientArray{T,2,A}

SparseArrays.nnz(a::TProductGradientSparseMatrix) = nnz(a.array)
SparseArrays.nzrange(a::TProductGradientSparseMatrix,col::Integer) = nzrange(a.array,col)
SparseArrays.rowvals(a::TProductGradientSparseMatrix) = rowvals(a.array)
SparseArrays.nonzeros(a::TProductGradientSparseMatrix) = a.array

"""
"""
struct MultiValueTProductArray{T,N,A<:AbstractTProductArray{T,N}} <: AbstractTProductArray{T,N}
  array::A
  ncomps::Int
end

function tproduct_array(
  array::A,
  arrays_1d::Vector{A},
  index_map::NTuple{N,I}
  ) where {T,N,D,Ti,A<:AbstractArray{T,N},I<:TProductIndexMap{D,Ti,<:AbstractMultiValueIndexMap}}

  array = TProductArray(array,arrays_1d,index_map)
  ncomps = num_components(first(index_map).indices)
  @check all(num_components(index_map[i].indices) for i = 2:N)
  MultiValueTProductArray(array,ncomps)
end

function tproduct_array(
  array::A,
  arrays_1d::Vector{A},
  gradients_1d::Vector{A},
  index_map::NTuple{N,I}
  ) where {T,N,D,Ti,A<:AbstractArray{T,N},I<:TProductIndexMap{D,Ti,<:AbstractMultiValueIndexMap}}

  array = TProductGradientArray(array,arrays_1d,gradients_1d,index_map)
  ncomps = num_components(first(index_map).indices)
  @check all(num_components(index_map[i].indices) for i = 2:N)
  MultiValueTProductArray(array,ncomps)
end

Arrays.get_array(a::MultiValueTProductArray) = get_array(a.array)
get_index_map(a::MultiValueTProductArray) = get_index_map(a.array)
univariate_length(a::MultiValueTProductArray) = univariate_length(a.array) + 1
TensorValues.num_components(a::MultiValueTProductArray) = a.ncomps

function IndexMaps.get_component(a::MultiValueTProductArray{T,N,<:TProductArray},ncomp::Integer) where {T,N}
  @check ncomp ≤ num_components(a)
  @unpack array,arrays_1d,index_map = a.array
  index_map′ = get_component(index_map,ncomp)
  tproduct_array(array,arrays_1d,index_map′)
end

function IndexMaps.get_component(a::MultiValueTProductArray{T,N,<:TProductGradientArray},ncomp::Integer) where {T,N}
  @check ncomp ≤ num_components(a)
  @unpack array,arrays_1d,gradients_1d,index_map = a.array
  index_map′ = get_component(index_map,ncomp)
  tproduct_array(array,arrays_1d,gradients_1d,index_map′)
end

const TProductMultiValueSparseMatrix{
  T,A<:Union{TProductSparseMatrix,TProductGradientSparseMatrix}
  } = MultiValueTProductArray{T,2,A}

SparseArrays.nnz(a::TProductMultiValueSparseMatrix) = nnz(a.array)
SparseArrays.nzrange(a::TProductMultiValueSparseMatrix,col::Integer) = nzrange(a.array,col)
SparseArrays.rowvals(a::TProductMultiValueSparseMatrix) = rowvals(a.array)
SparseArrays.nonzeros(a::TProductMultiValueSparseMatrix) = nonzeros(a.array)

# MultiField interface
struct BlockTProductArray{A<:AbstractTProductArray,N} <: AbstractArray{A,N}
  array::Array{A,N}
end

function tproduct_array(
  array::A,
  arrays_1d::Vector{A},
  index_map::NTuple{N,I}
  ) where {A<:BlockArray,N,I<:Vector{<:TProductIndexMap}}

  nblocks = blocklength(array)
  arrays = map(1:nblocks) do i
    array_i = blocks(array)[i]
    arrays_1d_i = map(blocks,arrays_1d)[i]
    index_map_i = getindex.(index_map,i)
    tproduct_array(array_i,arrays_1d_i,index_map_i)
  end
  BlockTProductArray(array)
end

function tproduct_array(
  array::A,
  arrays_1d::Vector{A},
  gradients_1d::Vector{A},
  index_map::NTuple{N,I}
  ) where {A<:BlockArray,N,I<:Vector{<:TProductIndexMap}}

  nblocks = blocklength(array)
  arrays = map(1:nblocks) do i
    array_i = blocks(array)[i]
    arrays_1d_i = map(blocks,arrays_1d)[i]
    gradients_1d_i = map(blocks,gradients_1d)[i]
    index_map_i = getindex.(index_map,i)
    tproduct_array(array_i,arrays_1d_i,gradients_1d_i,index_map_i)
  end
  BlockTProductArray(array)
end

Base.size(a::BlockTProductArray) = size(a.array)

Base.@propagate_inbounds function Base.getindex(
  a::BlockTProductArray{A,N},
  i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck blockcheckbounds(a.array,i)
  getindex(a.array,i...)
end

Base.@propagate_inbounds function Base.setindex!(
  a::BlockTProductArray{A,N},
  v,i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck blockcheckbounds(a.array,i)
  setindex!(a.array,v,i...)
end

# otherwise the sum/subtraction is a regular array
function Base.:+(A::AbstractBlockArray,B::AbstractBlockArray)
  @check axes(A) == axes(B)
  AB = (+)(A.blocks,B.blocks)
  BlockArrays._BlockArray(AB,A.axes)
end

function Base.:-(A::AbstractBlockArray,B::AbstractBlockArray)
  @check axes(A) == axes(B)
  AB = (-)(A.blocks,B.blocks)
  BlockArrays._BlockArray(AB,A.axes)
end
