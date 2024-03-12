struct TTIndexMap{D}
  # decide what to put here
end
TTIndexMap() = TTIndexMap{1}()
TTIndexMap(d::Integer) = TTIndexMap{d}()

struct TTArray{D,T,N,V} <: AbstractArray{T,N}
  values::V
  index_map::TTIndexMap{D}
  function TTArray(values::V,::Val{D}) where {T,N,V<:AbstractArray{T,N},D}
    new{D,T,N,V}(values,TTIndexMap(D))
  end
end

TTArray(values::AbstractArray) = TTArray(values,Val(1))

const TTVector{D,T,V} = TTArray{D,T,1,V}
const TTMatrix{D,T,V} = TTArray{D,T,2,V}
const TTSparseMatrix = TTArray{D,T,2,V} where {D,T,V<:AbstractSparseMatrix}
const TTSparseMatrixCSC = TTArray{D,T,2,V} where {D,T,V<:SparseMatrixCSC}

function TTVector{D,T}(::UndefInitializer,s) where {D,T}
  values = zeros(T,s)
  TTArray(values,Val(D))
end

function TTMatrix{D,T}(::UndefInitializer,s) where {D,T}
  values = zeros(T,s)
  TTArray(values,Val(D))
end

function TTSparseMatrixCSC{D,T}(::UndefInitializer,s) where {D,T}
  values = spzeros(T,s)
  TTArray(values,Val(D))
end

const ParamTTArray = ParamArray{T,N,A,L} where {T,N,A<:AbstractVector{<:TTArray},L}
const ParamTTVector = ParamArray{T,1,A,L} where {T,A<:AbstractVector{<:TTVector},L}
const ParamTTMatrix = ParamArray{T,2,A,L} where {T,A<:AbstractVector{<:TTMatrix},L}
const ParamTTSparseMatrix = ParamArray{T,2,A,L} where {T,A<:AbstractVector{<:TTSparseMatrix},L}
const ParamTTSparseMatrixCSC = ParamArray{T,2,A,L} where {T,A<:AbstractVector{<:TTSparseMatrixCSC},L}

Base.eltype(a::TTArray{D,T,N,V}) where {D,T,N,V} = T
Base.eltype(::Type{TTArray{D,T,N,V}}) where {D,T,N,V} = T
Base.ndims(a::TTArray{D,T,N,V}) where {D,T,N,V} = N
Base.ndims(::Type{TTArray{D,T,N,V}}) where {D,T,N,V} = N
Base.length(a::TTArray) = length(a.values)
Base.size(a::TTArray,i...) = size(a.values,i...)
Base.axes(a::TTArray,i...) = axes(a.values,i...)
Base.eachindex(a::TTArray) = eachindex(a.values)

Base.getindex(a::TTArray,i...) = getindex(a.values,i...)
Base.setindex!(a::TTArray,v,i...) = setindex!(a.values,v,i...)

Base.copy(a::TTArray{D}) where D = TTArray(copy(a.values),Val(D))
Base.copyto!(a::TTArray,b::TTArray) = copyto!(a.values,b.values)

function Base.similar(
  a::TTArray{D,T,N},
  element_type::Type{S}=T,
  dims::Tuple{Int,Vararg{Int}}=size(a)) where {D,T,S,N}
  TTArray(similar(a.values,element_type,dims),Val(D))
end

get_dim(a::TTArray{D}) where D = D
get_dim(::Type{<:TTArray{D}}) where D = D
get_dim(a::ParamTTArray{T,N,A}) where {T,N,A} = get_dim(eltype(A))
get_dim(::Type{<:ParamTTArray{T,N,A}}) where {T,N,A} = get_dim(eltype(A))

function Base.similar(::Type{TTArray{D,T,N,V}},n::Integer...) where {D,T,N,V}
  values = similar(V,n...)
  TTArray(values,Val(D))
end

function Base.sum(a::TTArray)
  sum(a.values)
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::TTArray{D},b::TTArray{D}) where D
      TTArray(($op)(a.values,b.values),Val(D))
    end
  end
end

(Base.:-)(a::TTArray) = a .* -1

function Base.:*(a::TTArray{D},b::Number) where D
  TTArray(a.values*b)
end

function Base.:*(a::Number,b::TTArray)
  b*a
end

function Base.:/(a::TTArray,b::Number)
  a*(1/b)
end

function Base.:*(a::TTMatrix{D},b::TTVector{D}) where D
  TTArray(a.values*b.values,Val(D))
end

function Base.:\(a::TTMatrix{D},b::TTVector{D}) where D
  TTArray(a.values\b.values,Val(D))
end

function Base.transpose(a::TTArray{D}) where D
  TTArray(transpose(a.values),Val(D))
end

function Base.fill!(a::TTArray,v)
  fill!(a.values,v)
  return a
end

function LinearAlgebra.fillstored!(a::TTSparseMatrix,v)
  LinearAlgebra.fillstored!(a.values,v)
  return a
end

function LinearAlgebra.mul!(c::TTArray,a::TTArray,b::TTArray,α::Number,β::Number)
  mul!(c.values,a.values,b.values,α,β)
  return c
end

function LinearAlgebra.ldiv!(a::TTArray,m::LU,b::TTArray)
  ldiv!(a.values,m,b.values)
  return a
end

function LinearAlgebra.rmul!(a::TTArray,b::Number)
  rmul!(a.values,b)
  return a
end

function LinearAlgebra.lu!(a::TTArray,b::TTArray)
  lu!(a.array,b.array)
  return a
end

function SparseArrays.resize!(a::TTArray,args...)
  resize!(a.array,args...)
  return a
end

function SparseArrays.sparse(
  I::AbstractVector,J::AbstractVector,V::TTVector{D},m::Integer,n::Integer) where D
  TTArray(sparse(I,J,V.values,m,n),Val(D))
end

SparseArrays.nnz(a::TTSparseMatrix) = nnz(a.values)
SparseArrays.findnz(a::TTSparseMatrix) = findnz(a.values)
SparseArrays.nzrange(a::TTSparseMatrix,col::Int) = nzrange(a.values,col)
SparseArrays.rowvals(a::TTSparseMatrix) = rowvals(a.values)
SparseArrays.nonzeros(a::TTSparseMatrix{D}) where D = TTArray(nonzeros(a.values),Val(D))
SparseMatricesCSR.colvals(a::TTSparseMatrix) = colvals(a.values)
SparseMatricesCSR.getoffset(a::TTSparseMatrix) = getoffset(a.values)

LinearAlgebra.cholesky(a::TTSparseMatrix) = cholesky(a.values)

function Arrays.CachedArray(a::TTArray)
  TTArray(CachedArray(a.values))
end

function Arrays.setsize!(
  a::TTArray{T,N,AbstractVector{CachedArray{T,N}}},
  s::NTuple{N,Int}) where {T,N}

  for ai in a
    setsize!(ai,s)
  end
  return a
end

function Arrays.SubVector(a::TTArray,pini::Int,pend::Int)
  TTArray(SubVector(a.values,pini,pend))
end

struct TTBroadcast{D,V}
  values::V
  TTBroadcast{D}(values::V) where {D,V} = new{D,V}(values)
end

_get_values(a::TTArray) = a.values
_get_values(a::TTBroadcast) = a.values

function Base.broadcasted(f,a::Union{TTArray{D},TTBroadcast{D}}...) where D
  TTBroadcast{D}(Base.broadcasted(f,map(_get_values,a)...))
end

function Base.broadcasted(f,a::Union{TTArray{D},TTBroadcast{D}},b::Number) where D
  TTBroadcast{D}(Base.broadcasted(f,_get_values(a),b))
end

function Base.broadcasted(f,a::Number,b::Union{TTArray,TTBroadcast{D}}) where D
  TTBroadcast{D}(Base.broadcasted(f,a,_get_values(b)))
end

function Base.broadcasted(f,
  a::Union{TTArray,TTBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{TTArray,TTBroadcast})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::TTBroadcast{D}) where D
  a = Base.materialize(_get_values(b))
  TTArray(a,Val(D))
end

function Base.materialize!(a::TTArray,b::Broadcast.Broadcasted)
  Base.materialize!(_get_values(a),b)
  a
end

function Base.materialize!(a::TTArray,b::TTBroadcast)
  Base.materialize!(_get_values(a),_get_values(b))
  a
end

Base.similar(a::TTBroadcast{D},args...) where D = TTBroadcast{D}(similar(a.values,args...))
Base.axes(a::TTBroadcast,i...) = axes(a.values,i...)
Base.eachindex(a::TTBroadcast) = eachindex(a.values)
Base.IndexStyle(a::TTBroadcast) = IndexStyle(a.values)
Base.ndims(a::TTBroadcast) = ndims(a.values)
Base.ndims(::Type{TTBroadcast{D,V}}) where {D,V} = ndims(V)
Base.size(a::TTBroadcast) = size(a.values)
Base.length(a::TTBroadcast) = length(a.values)
Base.iterate(a::TTBroadcast,i...) = iterate(a.values,i...)
