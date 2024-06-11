abstract type AbstractTProductArray{T,N} <: AbstractArray{T,N} end

Arrays.get_array(a::AbstractTProductArray) = @abstractmethod
get_index_map(a::AbstractTProductArray) = @abstractmethod

Base.size(a::AbstractTProductArray) = size(get_array(a))

Base.@propagate_inbounds _map_getindex(args...) = @abstractmethod
Base.@propagate_inbounds _map_getindex(imap::TProductIndexMap,i::Int) = (imap[i],)
Base.@propagate_inbounds _map_getindex(imap::TProductIndexMap,jmap::TProductIndexMap,i::Int,j::Int) = (imap[i],jmap[j])

Base.@propagate_inbounds function Base.getindex(a::AbstractTProductArray{T,N},i::Vararg{Integer,N}) where T
  i′ = _map_getindex(get_index_map(a)...,i...)
  getindex(a.array,i′...)
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

struct TProductArray{T,N,A,I} <: AbstractTProductArray{T,N}
  array::A
  arrays_1d::Vector{A}
  index_map::NTuple{N,I}
  function TProductArray(
    array::A,
    arrays_1d::Vector{A},
    index_map::NTuple{N,I}
    ) where {T,N,A<:AbstractArray{T,N},I<:TProductIndexMap}
    new{T,N,A,I}(array,arrays_1d,index_map)
  end
end

Arrays.get_array(a::TProductArray) = a.array

function TProductArray(arrays_1d::Vector{A},index_map::NTuple) where A
  array::A = _kron(arrays_1d...)
  TProductArray(array,arrays_1d,index_map)
end

function Base.similar(a::TProductArray{T,N},::Type{T′},dims::Dims{N}) where {T,T′,N}
  TProductArray(similar(a.array,T′,dims...),a.arrays_1d,a.index_map)
end

function Base.copyto!(a::TProductArray,b::TProductArray)
  @check size(a) == size(b)
  copyto!(a.array,a.array)
  map(copyto!,a.arrays_1d,b.arrays_1d)
  map(copyto!,a.index_map,b.index_map)
  a
end

const TProductSparseMatrix = TProductArray{T,2,A} where {T,A<:AbstractSparseMatrix}

SparseArrays.nnz(a::TProductSparseMatrix) = nnz(a.array)
SparseArrays.nzrange(a::TProductSparseMatrix,col::Integer) = nzrange(a.array,col)
SparseArrays.rowvals(a::TProductSparseMatrix) = rowvals(a.array)
SparseArrays.nonzeros(a::TProductSparseMatrix) = a.array

function _kron(A::AbstractArray...)
  kron(reverse(A)...)
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

struct TProductGradientArray{T,N,A,I} <: AbstractTProductArray{T,N}
  array::A
  arrays_1d::Vector{A}
  gradients_1d::Vector{A}
  index_map::NTuple{N,I}
  function TProductGradientArray(
    array::A,
    arrays_1d::Vector{A},
    gradients_1d::Vector{A},
    index_map::NTuple{N,I}
    ) where {T,N,A<:AbstractArray{T,N},I<:TProductIndexMap}

    new{T,N,A,I}(array,arrays_1d,gradients_1d,index_map)
  end
end

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
  map(copyto!,a.index_map,b.index_map)
  a
end

const TProductGradientSparseMatrix{T,A<:AbstractSparseMatrix} = TProductGradientArray{T,2,A}

SparseArrays.nnz(a::TProductGradientSparseMatrix) = nnz(a.array)
SparseArrays.nzrange(a::TProductGradientSparseMatrix,col::Integer) = nzrange(a.array,col)
SparseArrays.rowvals(a::TProductGradientSparseMatrix) = rowvals(a.array)
SparseArrays.nonzeros(a::TProductGradientSparseMatrix) = a.array

# deal with cell field + gradient cell field
for op in (:+,:-)
  @eval ($op)(a::Vector,b::TProductGradientEval) = TProductGradientEval(b.f,b.g,$op)
  @eval ($op)(a::TProductGradientEval,b::Vector) = TProductGradientEval(a.f,a.g,$op)
end
