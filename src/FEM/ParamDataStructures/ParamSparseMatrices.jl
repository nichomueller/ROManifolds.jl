ParamArray(A::AbstractVector{<:SparseMatrixCSC};kwargs...) = ConsecutiveParamSparseMatrixCSC(A)
ParamArray(A::AbstractVector{<:SparseMatrixCSR};kwargs...) = ConsecutiveParamSparseMatrixCSR(A)

function SparseArrays.sparse(
  m::Int,
  n::Int,
  colptr::Vector{<:Integer},
  rowval::Vector{<:Integer},
  nzval::ConsecutiveParamVector,
  combine=+)

  nzdata = get_all_data(nzval)
  ConsecutiveParamSparseMatrixCSC(m,n,colptr,rowval,nzdata)
end

function SparseMatricesCSR.sparsecsr(
  v::Val{Bi},
  m::Int,
  n::Int,
  rowptr::Vector{<:Integer},
  colval::Vector{<:Integer},
  nzval::ConsecutiveParamVector,
  combine=+) where Bi

  nzdata = get_all_data(nzval)
  ConsecutiveParamSparseMatrixCSR{Bi}(v,m,n,rowptr,colval,nzdata)
end

Base.size(A::ParamSparseMatrix) = (param_length(A),param_length(A))

Base.@propagate_inbounds function Base.setindex!(A::ParamSparseMatrix,v,i::Integer,j::Integer)
  @notimplemented
end

get_all_data(A::ParamSparseMatrix) = @abstractmethod

function LinearAlgebra.fillstored!(A::ParamSparseMatrix,b::Number)
  fill!(get_all_data(A),b)
  return A
end

# small hack, we shouldn't be able to fill an abstract array with a non-scalar
function LinearAlgebra.fillstored!(A::ParamSparseMatrix,b::AbstractMatrix{<:Number})
  @check all(b.==first(b))
  LinearAlgebra.fillstored!(A,first(b))
  return A
end

function Base.fill!(A::ParamSparseMatrix,b::Number)
  fill!(get_all_data(A),b)
  return A
end

function LinearAlgebra.rmul!(A::ParamSparseMatrix,b::Number)
  rmul!(get_all_data(A),b)
  return A
end

function LinearAlgebra.axpy!(α::Number,A::ParamSparseMatrix,B::ParamSparseMatrix)
  axpy!(α,get_all_data(A),get_all_data(B))
  return B
end

# small hack
Base.iszero(A::ParamSparseMatrix) = (nnz(A) == 0)

function DofMaps.recast(a::AbstractMatrix,A::AbstractSparseMatrix)
  @check size(a,1) == nnz(A)
  B = map(v -> recast(v,A),collect.(eachcol(a)))
  return ParamArray(B)
end

function DofMaps.recast(a::AbstractArray,A::ParamSparseMatrix)
  recast(a,param_getindex(A,1))
end

"""
    abstract type ParamSparseMatrixCSC{Tv,Ti} <: ParamSparseMatrix{Tv,Ti,SparseMatrixCSC{Tv,Ti}} end

Type representing parametric sparse matrices in CSC format.
Subtypes:
- [`ConsecutiveParamSparseMatrixCSC`](@ref)
- [`GenericParamSparseMatrixCSC`](@ref)
"""
abstract type ParamSparseMatrixCSC{Tv,Ti} <: ParamSparseMatrix{Tv,Ti,SparseMatrixCSC{Tv,Ti}} end

"""
    struct ConsecutiveParamSparseMatrixCSC{Tv,Ti<:Integer} <: ParamSparseMatrixCSC{Tv,Ti}
      m::Int64
      n::Int64
      colptr::Vector{Ti}
      rowval::Vector{Ti}
      data::Matrix{Tv}
    end

Represents a vector of sparse matrices in CSC format, with entries stored
consecutively in memory. For sake of coherence, an instance of
`ConsecutiveParamSparseMatrixCSC` inherits from AbstractMatrix{<:SparseMatrixCSC{Tv,Ti}
rather than AbstractVector{<:SparseMatrixCSC{Tv,Ti}, but should conceptually be
thought as an AbstractVector{<:SparseMatrixCSC{Tv,Ti}.
"""
struct ConsecutiveParamSparseMatrixCSC{Tv,Ti<:Integer} <: ParamSparseMatrixCSC{Tv,Ti}
  m::Int64
  n::Int64
  colptr::Vector{Ti}
  rowval::Vector{Ti}
  data::Matrix{Tv}
end

param_length(A::ConsecutiveParamSparseMatrixCSC) = size(A.data,2)
get_all_data(A::ConsecutiveParamSparseMatrixCSC) = A.data

MemoryLayoutStyle(::Type{<:ConsecutiveParamSparseMatrixCSC}) = ConsecutiveMemory()

SparseArrays.getcolptr(A::ConsecutiveParamSparseMatrixCSC) = A.colptr
SparseArrays.rowvals(A::ConsecutiveParamSparseMatrixCSC) = A.rowval
SparseArrays.nonzeros(A::ConsecutiveParamSparseMatrixCSC) = ConsecutiveParamArray(A.data)
SparseArrays.nnz(A::ConsecutiveParamSparseMatrixCSC) = Int(getcolptr(A)[innersize(A,2)+1])-1

function ConsecutiveParamSparseMatrixCSC(a::AbstractVector{<:SparseMatrixCSC{Tv}}) where Tv
  item = testitem(a)
  m,n = size(item)
  colptr = getcolptr(item)
  rowval = rowvals(item)

  @notimplementedif (any(a->size(a) != (m,n),a))

  if (any(a->getcolptr(a) != colptr, a) || any(a->rowvals(a) != rowval, a))
    GenericParamSparseMatrixCSC(a)
  end

  ndata = nnz(item)
  plength = length(a)
  data = Matrix{Tv}(undef,ndata,plength)
  p = 1
  @inbounds for i in 1:plength
    ai = a[i]
    for j in 1:ndata
      aij = nonzeros(ai)[j]
      data[p] = aij
      p += 1
    end
  end

  ConsecutiveParamSparseMatrixCSC(m,n,colptr,rowval,data)
end

function param_array(a::SparseMatrixCSC,l::Integer;kwargs...)
  outer = (1,1,l)
  data = repeat(nonzeros(a);outer)
  !copy && LinearAlgebra.fillstored!(data,zero(eltype(a)))
  m,n = size(a)
  colptr = getcolptr(a)
  rowval = rowvals(a)
  ConsecutiveParamSparseMatrixCSC(m,n,colptr,rowval,data)
end

ArraysOfArrays.innersize(A::ConsecutiveParamSparseMatrixCSC) = (A.m,A.n)

Base.@propagate_inbounds function Base.getindex(A::ConsecutiveParamSparseMatrixCSC,i::Integer,j::Integer)
  @boundscheck checkbounds(A,i,j)
  if i == j
    SparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,getindex(A.data,:,i))
  else
    spzeros(innersize(A))
  end
end

Base.@propagate_inbounds function Base.setindex!(
  A::ConsecutiveParamSparseMatrixCSC,v::SparseMatrixCSC,i::Integer,j::Integer)

  @boundscheck checkbounds(A,i,j)
  if i == j
    @assert innersize(A)==size(v) && getcolptr(A)==getcolptr(v) && rowvals(A)==rowvals(v)
    setindex!(A.data,nonzeros(v),:,i)
  end
  v
end

function Base.similar(A::ConsecutiveParamSparseMatrixCSC)
  ConsecutiveParamSparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,similar(A.data))
end

function Base.copy(A::ConsecutiveParamSparseMatrixCSC)
  ConsecutiveParamSparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,copy(A.data))
end

function Base.copyto!(A::ConsecutiveParamSparseMatrixCSC,B::ConsecutiveParamSparseMatrixCSC)
  @check size(A)==size(B)
  copyto!(A.colptr,B.colptr)
  copyto!(A.rowval,B.rowval)
  copyto!(A.data,B.data)
end

"""
    struct GenericParamSparseMatrixCSC{Tv,Ti<:Integer} <: ParamSparseMatrixCSC{Tv,Ti}
      m::Int64
      n::Int64
      colptr::Vector{Ti}
      rowval::Vector{Ti}
      data::Vector{Tv}
      ptrs::Vector{Ti}
    end

Represents a vector of sparse matrices in CSC format, with entries stored
non-consecutively in memory. For sake of coherence, an instance of
`GenericParamSparseMatrixCSC` inherits from AbstractMatrix{<:SparseMatrixCSC{Tv,Ti}
rather than AbstractVector{<:SparseMatrixCSC{Tv,Ti}, but should conceptually be
thought as an AbstractVector{<:SparseMatrixCSC{Tv,Ti}.
"""
struct GenericParamSparseMatrixCSC{Tv,Ti<:Integer} <: ParamSparseMatrixCSC{Tv,Ti}
  m::Int64
  n::Int64
  colptr::Vector{Ti}
  rowval::Vector{Ti}
  data::Vector{Tv}
  ptrs::Vector{Ti}
end

param_length(A::GenericParamSparseMatrixCSC) = length(A.ptrs)-1
get_all_data(A::GenericParamSparseMatrixCSC) = A.data

function GenericParamSparseMatrixCSC(a::AbstractVector{<:SparseMatrixCSC{Tv}}) where Tv
  item = testitem(a)
  m,n = size(item)
  plength = length(a)
  ptrs = _vec_of_pointers(a)
  Ti = eltype(ptrs)
  u = one(Ti)
  ndata = ptrs[end]-u
  colptr = Vector{Ti}(undef,ndata)
  rowval = Vector{Ti}(undef,ndata)
  data = Vector{Tv}(undef,ndata)
  p = 1
  @inbounds for i in 1:plength
    ai = a[i]
    for j in nnz(ai)
      colptr[p] = getcolptr(ai)[j]
      rowval[p] = rowvals(ai)[j]
      data[p] = nonzeros(ai)[j]
      p += 1
    end
  end
  GenericParamSparseMatrixCSC(m,n,colptr,rowval,data,ptrs)
end

ArraysOfArrays.innersize(A::GenericParamSparseMatrixCSC) = (A.m,A.n)

Base.@propagate_inbounds function Base.getindex(A::GenericParamSparseMatrixCSC{Tv},i::Integer,j::Integer) where Tv
  @boundscheck checkbounds(A,i,j)
  if i == j
    u = one(eltype(A.ptrs))
    pini = A.ptrs[i]
    pend = A.ptrs[i+1]-u
    colptr = A.colptr[pini:pend]
    rowval = A.rowval[pini:pend]
    data = A.data[pini:pend]
    SparseMatrixCSC(A.m,A.n,colptr,rowval,data)
  else
    fill(zero(Tv),nrow,ncol)
  end
end

Base.@propagate_inbounds function Base.setindex!(
  A::GenericParamSparseMatrixCSC,v::SparseMatrixCSC,i::Integer,j::Integer)

  @boundscheck checkbounds(A,i,j)
  if i == j
    @assert innersize(A)==size(v)
    u = one(eltype(A.ptrs))
    pini = A.ptrs[i]
    pend = A.ptrs[i+1]-u
    colptr = A.colptr[pini:pend]
    rowval = A.rowval[pini:pend]
    data = A.data[pini:pend]
    copyto!(colptr,getcolptr(v))
    copyto!(rowval,rowvals(v))
    copyto!(data,nonzeros(v))
  end
  v
end

function Base.similar(A::GenericParamSparseMatrixCSC)
  GenericParamSparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,similar(A.data),A.ptrs)
end

function Base.copy(A::GenericParamSparseMatrixCSC)
  GenericParamSparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,copy(A.data),A.ptrs)
end

function Base.copyto!(A::GenericParamSparseMatrixCSC,B::GenericParamSparseMatrixCSC)
  @check size(A)==size(B)
  copyto!(A.colptr,B.colptr)
  copyto!(A.rowval,B.rowval)
  copyto!(A.data,B.data)
  copyto!(A.ptrs,B.ptrs)
end

# CSR FORMAT

"""
    abstract type ParamSparseMatrixCSR{Bi,Tv,Ti} <: ParamSparseMatrix{Tv,Ti,SparseMatrixCSR{Bi,Tv,Ti}} end

Type representing parametric sparse matrices in CSR format.
Subtypes:
- [`ConsecutiveParamSparseMatrixCSR`](@ref)
- [`GenericParamSparseMatrixCSR`](@ref)
"""
abstract type ParamSparseMatrixCSR{Bi,Tv,Ti} <: ParamSparseMatrix{Tv,Ti,SparseMatrixCSR{Bi,Tv,Ti}} end

"""
    struct ConsecutiveParamSparseMatrixCSR{Bi,Tv,Ti<:Integer} <: ParamSparseMatrixCSR{Bi,Tv,Ti}
      m::Int64
      n::Int64
      rowptr::Vector{Ti}
      colval::Vector{Ti}
      data::Matrix{Tv}
    end

Represents a vector of sparse matrices in CSR format, with entries stored
consecutively in memory. For sake of coherence, an instance of
`ConsecutiveParamSparseMatrixCSR` inherits from AbstractMatrix{<:SparseMatrixCSR{Bi,Tv,Ti}
rather than AbstractVector{<:SparseMatrixCSR{Bi,Tv,Ti}, but should conceptually be
thought as an AbstractVector{<:SparseMatrixCSR{Bi,Tv,Ti}.
"""
struct ConsecutiveParamSparseMatrixCSR{Bi,Tv,Ti<:Integer} <: ParamSparseMatrixCSR{Bi,Tv,Ti}
  m::Int64
  n::Int64
  rowptr::Vector{Ti}
  colval::Vector{Ti}
  data::Matrix{Tv}
  function ConsecutiveParamSparseMatrixCSR{Bi}(
    m::Int64,
    n::Int64,
    rowptr::Vector{Ti},
    colval::Vector{Ti},
    data::Matrix{Tv}
    ) where {Bi,Tv,Ti}

    new{Bi,Tv,Ti}(m,n,rowptr,colval,data)
  end
end

param_length(A::ConsecutiveParamSparseMatrixCSR) = size(A.data,2)
get_all_data(A::ConsecutiveParamSparseMatrixCSR) = A.data

MemoryLayoutStyle(::Type{<:ConsecutiveParamSparseMatrixCSR}) = ConsecutiveMemory()

SparseMatricesCSR.getrowptr(A::ConsecutiveParamSparseMatrixCSR) = A.rowptr
SparseMatricesCSR.colvals(A::ConsecutiveParamSparseMatrixCSR) = A.colval
SparseArrays.nonzeros(A::ConsecutiveParamSparseMatrixCSR) = ConsecutiveParamArray(A.data)
SparseArrays.nnz(A::ConsecutiveParamSparseMatrixCSR) = size(A.data,1)

function ConsecutiveParamSparseMatrixCSR(a::AbstractVector{<:SparseMatrixCSR{Bi,Tv}}) where {Bi,Tv}
  item = testitem(a)
  m,n = size(item)
  rowptr = getrowptr(item)
  colval = colvals(item)

  @notimplementedif (any(a->size(a) != (m,n),a))

  if (any(a->getrowptr(a) != rowptr, a) || any(a->colvals(a) != colval, a))
    GenericParamSparseMatrixCSR(a)
  end

  ndata = nnz(item)
  plength = length(a)
  data = Matrix{Tv}(undef,ndata,plength)
  p = 1
  @inbounds for i in 1:plength
    ai = a[i]
    for j in 1:ndata
      aij = nonzeros(ai)[j]
      data[p] = aij
      p += 1
    end
  end

  ConsecutiveParamSparseMatrixCSR{1}(m,n,rowptr,colval,data)
end

function param_array(a::SparseMatrixCSR,l::Integer;kwargs...)
  outer = (1,1,l)
  data = repeat(nonzeros(a);outer)
  !copy && LinearAlgebra.fillstored!(data,zero(eltype(a)))
  m,n = size(a)
  rowptr = getrowptr(a)
  colval = colvals(a)
  ConsecutiveParamSparseMatrixCSR{1}(m,n,rowptr,colval,data)
end

ArraysOfArrays.innersize(A::ConsecutiveParamSparseMatrixCSR) = (A.m,A.n)

Base.@propagate_inbounds function Base.getindex(A::ConsecutiveParamSparseMatrixCSR{Bi},i::Integer,j::Integer) where Bi
  @boundscheck checkbounds(A,i,j)
  if i == j
    SparseMatrixCSR{Bi}(A.m,A.n,A.rowptr,A.colval,getindex(A.data,:,i))
  else
    spzeros(innersize(A))
  end
end

Base.@propagate_inbounds function Base.setindex!(
  A::ConsecutiveParamSparseMatrixCSR,v::SparseMatrixCSR,i::Integer,j::Integer)

  @boundscheck checkbounds(A,i,j)
  if i == j
    @assert innersize(A)==size(v) && getrowptr(A)==getrowptr(v) && colvals(A)==colvals(v)
    setindex!(A.data,nonzeros(v),:,i)
  end
  v
end

function Base.similar(A::ConsecutiveParamSparseMatrixCSR{Bi}) where Bi
  ConsecutiveParamSparseMatrixCSR{Bi}(A.m,A.n,A.rowptr,A.colval,similar(A.data))
end

function Base.copy(A::ConsecutiveParamSparseMatrixCSR{Bi}) where Bi
  ConsecutiveParamSparseMatrixCSR{Bi}(A.m,A.n,A.rowptr,A.colval,copy(A.data))
end

function Base.copyto!(A::ConsecutiveParamSparseMatrixCSR,B::ConsecutiveParamSparseMatrixCSR)
  @check size(A)==size(B)
  copyto!(A.rowptr,B.rowptr)
  copyto!(A.colval,B.colval)
  copyto!(A.data,B.data)
end

"""
    struct GenericParamSparseMatrixCSR{Bi,Tv,Ti<:Integer} <: ParamSparseMatrixCSR{Bi,Tv,Ti}
      m::Int64
      n::Int64
      rowptr::Vector{Ti}
      colval::Vector{Ti}
      data::Vector{Tv}
      ptrs::Vector{Ti}
    end

Represents a vector of sparse matrices in CSR format, with entries stored
non-consecutively in memory. For sake of coherence, an instance of
`GenericParamSparseMatrixCSR` inherits from AbstractMatrix{<:SparseMatrixCSR{Bi,Tv,Ti}
rather than AbstractVector{<:SparseMatrixCSR{Bi,Tv,Ti}, but should conceptually be
thought as an AbstractVector{<:SparseMatrixCSR{Bi,Tv,Ti}.
"""
struct GenericParamSparseMatrixCSR{Bi,Tv,Ti<:Integer} <: ParamSparseMatrixCSR{Bi,Tv,Ti}
  m::Int64
  n::Int64
  rowptr::Vector{Ti}
  colval::Vector{Ti}
  data::Vector{Tv}
  ptrs::Vector{Ti}
  function GenericParamSparseMatrixCSR{Bi}(
    m::Int64,
    n::Int64,
    rowptr::Vector{Ti},
    colval::Vector{Ti},
    data::Vector{Tv},
    ptrs::Vector{Ti}
    ) where {Bi,Tv,Ti}

    new{Bi,Tv,Ti}(m,n,rowptr,colval,data)
  end
end

param_length(A::GenericParamSparseMatrixCSR) = length(A.ptrs)-1
get_all_data(A::GenericParamSparseMatrixCSR) = A.data

function GenericParamSparseMatrixCSR(a::AbstractVector{<:SparseMatrixCSR{Bi,Tv}}) where {Bi,Tv}
  item = testitem(a)
  m,n = size(item)
  plength = length(a)
  ptrs = _vec_of_pointers(a)
  Ti = eltype(ptrs)
  u = one(Ti)
  ndata = ptrs[end]-u
  rowptr = Vector{Ti}(undef,ndata)
  colval = Vector{Ti}(undef,ndata)
  data = Vector{Tv}(undef,ndata)
  p = 1
  @inbounds for i in 1:plength
    ai = a[i]
    for j in nnz(ai)
      rowptr[p] = getrowptr(ai)[j]
      colval[p] = colvals(ai)[j]
      data[p] = nonzeros(ai)[j]
      p += 1
    end
  end
  GenericParamSparseMatrixCSR{1}(m,n,rowptr,colval,data,ptrs)
end

ArraysOfArrays.innersize(A::GenericParamSparseMatrixCSR) = (A.m,A.n)

Base.@propagate_inbounds function Base.getindex(A::GenericParamSparseMatrixCSR{Bi,Tv},i::Integer,j::Integer) where {Bi,Tv}
  @boundscheck checkbounds(A,i,j)
  if i == j
    u = one(eltype(A.ptrs))
    pini = A.ptrs[i]
    pend = A.ptrs[i+1]-u
    rowptr = A.rowptr[pini:pend]
    colval = A.colval[pini:pend]
    data = A.data[pini:pend]
    SparseMatrixCSR{Bi}(A.m,A.n,rowptr,colval,data)
  else
    fill(zero(Tv),nrow,ncol)
  end
end

Base.@propagate_inbounds function Base.setindex!(
  A::GenericParamSparseMatrixCSR,v::SparseMatrixCSR,i::Integer,j::Integer)

  @boundscheck checkbounds(A,i,j)
  if i == j
    @assert innersize(A)==size(v)
    u = one(eltype(A.ptrs))
    pini = A.ptrs[i]
    pend = A.ptrs[i+1]-u
    rowptr = A.rowptr[pini:pend]
    colval = A.colval[pini:pend]
    data = A.data[pini:pend]
    copyto!(rowptr,getrowptr(v))
    copyto!(colval,colvals(v))
    copyto!(data,nonzeros(v))
  end
  v
end

function Base.similar(A::GenericParamSparseMatrixCSR{Bi}) where Bi
  GenericParamSparseMatrixCSC{Bi}(A.m,A.n,A.rowptr,A.colval,similar(A.data),A.ptrs)
end

function Base.copy(A::GenericParamSparseMatrixCSR{Bi}) where Bi
  GenericParamSparseMatrixCSC{Bi}(A.m,A.n,A.rowptr,A.colval,copy(A.data),A.ptrs)
end

function Base.copyto!(A::GenericParamSparseMatrixCSR,B::GenericParamSparseMatrixCSR)
  @check size(A)==size(B)
  copyto!(A.rowptr,B.rowptr)
  copyto!(A.colval,B.colval)
  copyto!(A.data,B.data)
  copyto!(A.ptrs,B.ptrs)
end

# utils

function _vec_of_pointers(a::AbstractVector{<:AbstractSparseMatrix})
  n = length(a)
  ptrs = Vector{Int}(undef,n+1)
  @inbounds for i in 1:n
    ai = a[i]
    ptrs[i+1] = nnz(ai)
  end
  length_to_ptrs!(ptrs)
  ptrs
end

const ConsecutiveParamSparseMatrix{Tv,Ti} = Union{
  ConsecutiveParamSparseMatrixCSC{Tv,Ti},
  ConsecutiveParamSparseMatrixCSR{<:Any,Tv,Ti}}
