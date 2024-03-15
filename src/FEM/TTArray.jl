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

get_values(a::TTArray) = a.values
get_values(a::ParamTTArray) = ParamArray(map(get_values,get_array(a)))

Base.eltype(a::TTArray{D,T,N,V}) where {D,T,N,V} = T
Base.eltype(::Type{TTArray{D,T,N,V}}) where {D,T,N,V} = T
Base.ndims(a::TTArray{D,T,N,V}) where {D,T,N,V} = N
Base.ndims(::Type{TTArray{D,T,N,V}}) where {D,T,N,V} = N
Base.length(a::TTArray) = length(get_values(a))
Base.size(a::TTArray,i...) = size(get_values(a),i...)
Base.axes(a::TTArray,i...) = axes(get_values(a),i...)
Base.eachindex(a::TTArray) = eachindex(get_values(a))

Base.getindex(a::TTArray,i...) = getindex(get_values(a),i...)
Base.setindex!(a::TTArray,v,i...) = setindex!(get_values(a),v,i...)

Base.copy(a::TTArray{D}) where D = TTArray(copy(get_values(a)),Val(D))
Base.copyto!(a::TTArray,b::TTArray) = copyto!(get_values(a),get_values(b))

function Base.similar(
  a::TTArray{D,T,N},
  element_type::Type{S}=T,
  dims::Tuple{Int,Vararg{Int}}=size(a)) where {D,T,S,N}
  TTArray(similar(get_values(a),element_type,dims),Val(D))
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
  sum(get_values(a))
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::TTArray{D},b::TTArray{D}) where D
      TTArray(($op)(get_values(a),get_values(b)),Val(D))
    end
  end
end

(Base.:-)(a::TTArray) = a .* -1

function Base.:*(a::TTArray{D},b::Number) where D
  TTArray(get_values(a)*b)
end

function Base.:*(a::Number,b::TTArray)
  b*a
end

function Base.:/(a::TTArray,b::Number)
  a*(1/b)
end

function Base.:*(a::TTMatrix{D},b::TTVector{D}) where D
  TTArray(get_values(a)*get_values(b),Val(D))
end

function Base.:\(a::TTMatrix{D},b::TTVector{D}) where D
  TTArray(get_values(a)\get_values(b),Val(D))
end

function Base.transpose(a::TTArray{D}) where D
  TTArray(transpose(get_values(a)),Val(D))
end

function Base.fill!(a::TTArray,v)
  fill!(get_values(a),v)
  return a
end

function LinearAlgebra.fillstored!(a::TTSparseMatrix,v)
  LinearAlgebra.fillstored!(get_values(a),v)
  return a
end

function LinearAlgebra.mul!(c::TTArray,a::TTArray,b::TTArray,α::Number,β::Number)
  mul!(get_values(c),get_values(a),get_values(b),α,β)
  return c
end

function LinearAlgebra.ldiv!(a::TTArray,m::LU,b::TTArray)
  ldiv!(get_values(a),m,get_values(b))
  return a
end

function LinearAlgebra.rmul!(a::TTArray,b::Number)
  rmul!(get_values(a),b)
  return a
end

function LinearAlgebra.lu!(a::TTArray,b::TTArray)
  lu!(get_values(a),get_values(b))
  return a
end

function SparseArrays.resize!(a::TTArray,args...)
  resize!(get_values(a),args...)
  return a
end

function SparseArrays.sparse(
  I::AbstractVector,J::AbstractVector,V::TTVector{D},m::Integer,n::Integer) where D
  TTArray(sparse(I,J,get_values(V),m,n),Val(D))
end

SparseArrays.nnz(a::TTSparseMatrix) = nnz(get_values(a))
SparseArrays.findnz(a::TTSparseMatrix) = findnz(get_values(a))
SparseArrays.nzrange(a::TTSparseMatrix,col::Int) = nzrange(get_values(a),col)
SparseArrays.rowvals(a::TTSparseMatrix) = rowvals(get_values(a))
SparseArrays.nonzeros(a::TTSparseMatrix{D}) where D = TTArray(nonzeros(get_values(a)),Val(D))
SparseMatricesCSR.colvals(a::TTSparseMatrix) = colvals(get_values(a))
SparseMatricesCSR.getoffset(a::TTSparseMatrix) = getoffset(get_values(a))

LinearAlgebra.cholesky(a::TTSparseMatrix) = cholesky(get_values(a))

function Arrays.CachedArray(a::TTArray)
  TTArray(CachedArray(get_values(a)))
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
  TTArray(SubVector(get_values(a),pini,pend))
end

struct TTBroadcast{D,V}
  values::V
  TTBroadcast{D}(values::V) where {D,V} = new{D,V}(values)
end

get_values(a::TTBroadcast) = a.values

function Base.broadcasted(f,a::Union{TTArray{D},TTBroadcast{D}}...) where D
  TTBroadcast{D}(Base.broadcasted(f,map(get_values,a)...))
end

function Base.broadcasted(f,a::Union{TTArray{D},TTBroadcast{D}},b::Number) where D
  TTBroadcast{D}(Base.broadcasted(f,get_values(a),b))
end

function Base.broadcasted(f,a::Number,b::Union{TTArray,TTBroadcast{D}}) where D
  TTBroadcast{D}(Base.broadcasted(f,a,get_values(b)))
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
  a = Base.materialize(get_values(b))
  TTArray(a,Val(D))
end

function Base.materialize!(a::TTArray,b::Broadcast.Broadcasted)
  Base.materialize!(get_values(a),b)
  a
end

function Base.materialize!(a::TTArray,b::TTBroadcast)
  Base.materialize!(get_values(a),get_values(b))
  a
end

Base.similar(a::TTBroadcast{D},args...) where D = TTBroadcast{D}(similar(get_values(a),args...))
Base.axes(a::TTBroadcast,i...) = axes(get_values(a),i...)
Base.eachindex(a::TTBroadcast) = eachindex(get_values(a))
Base.IndexStyle(a::TTBroadcast) = IndexStyle(get_values(a))
Base.ndims(a::TTBroadcast) = ndims(get_values(a))
Base.ndims(::Type{TTBroadcast{D,V}}) where {D,V} = ndims(V)
Base.size(a::TTBroadcast) = size(get_values(a))
Base.length(a::TTBroadcast) = length(get_values(a))
Base.iterate(a::TTBroadcast,i...) = iterate(get_values(a),i...)
