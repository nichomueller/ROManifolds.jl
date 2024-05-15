struct TTArray{D,T,N,V,I} <: AbstractArray{T,N}
  values::V
  index_map::I
  function TTArray(values::V,index_map::I) where {T,N,D,V<:AbstractArray{T,N},I<:AbstractIndexMap{D}}
    new{D,T,N,V,I}(values,index_map)
  end
end

const TTVector{D,T,V} = TTArray{D,T,1,V}
const TTMatrix{D,T,V} = TTArray{D,T,2,V}
const TTSparseMatrix{D,T,V<:AbstractSparseMatrix} = TTArray{D,T,2,V}
const TTSparseMatrixCSC{D,T,V<:SparseMatrixCSC} = TTArray{D,T,2,V}

get_values(a::TTArray) = a.values
get_index_map(a::TTArray) = a.index_map

function TTVector{D,T,V,I}(::UndefInitializer,s,index_map) where {D,T,V,I}
  values = zeros(T,s)
  TTArray(values,index_map)
end

function TTMatrix{D,T,V,I}(::UndefInitializer,s,index_map) where {D,T,V,I}
  values = zeros(T,s)
  TTArray(values,index_map)
end

# without index map, we default to normal arrays

function TTVector{D,T,V,I}(::UndefInitializer,s) where {D,T,V,I}
  Vector{T}(undef,s)
end

function TTMatrix{D,T,V,I}(::UndefInitializer,s) where {D,T,V,I}
  Matrix{T}(undef,s)
end

const ParamTTArray = ParamArray{T,N,A,L} where {T,N,A<:AbstractVector{<:TTArray},L}
const ParamTTVector = ParamArray{T,1,A,L} where {T,A<:AbstractVector{<:TTVector},L}
const ParamTTMatrix = ParamArray{T,2,A,L} where {T,A<:AbstractVector{<:TTMatrix},L}
const ParamTTSparseMatrix = ParamArray{T,2,A,L} where {T,A<:AbstractVector{<:TTSparseMatrix},L}
const ParamTTSparseMatrixCSC = ParamArray{T,2,A,L} where {T,A<:AbstractVector{<:TTSparseMatrixCSC},L}

get_values(a::ParamTTArray) = ParamArray(map(get_values,get_array(a)))

Base.eltype(a::TTArray{D,T}) where {D,T} = T
Base.eltype(::Type{<:TTArray{D,T}}) where {D,T} = T
Base.ndims(a::TTArray{D,T,N}) where {D,T,N} = N
Base.ndims(::Type{<:TTArray{D,T,N}}) where {D,T,N} = N
Base.length(a::TTArray) = length(get_values(a))
Base.size(a::TTArray,i...) = size(get_values(a),i...)
Base.axes(a::TTArray,i...) = axes(get_values(a),i...)
Base.eachindex(a::TTArray) = eachindex(get_values(a))

Base.getindex(a::TTArray,i::Integer...) = getindex(get_values(a),i...)
Base.getindex(a::TTArray,i...) = TTArray(getindex(get_values(a),i...),get_index_map(a))
Base.setindex!(a::TTArray,v,i...) = setindex!(get_values(a),v,i...)

Base.copy(a::TTArray) = TTArray(copy(get_values(a)),get_index_map(a))
Base.copyto!(a::TTArray,b::TTArray) = copyto!(get_values(a),get_values(b))

function Base.similar(
  a::TTArray{D,T},
  element_type::Type{S}=T,
  dims::Tuple{Int,Vararg{Int}}=size(a)) where {D,T,S}
  TTArray(similar(get_values(a),element_type,dims),get_index_map(a))
end

get_dim(a::TTArray{D}) where D = D
get_dim(::Type{<:TTArray{D}}) where D = D
get_dim(a::ParamTTArray{T,N,A}) where {T,N,A} = get_dim(eltype(A))
get_dim(::Type{<:ParamTTArray{T,N,A}}) where {T,N,A} = get_dim(eltype(A))

function Base.similar(::Type{TTArray{D,T,N,V}},n::Integer...) where {D,T,N,V}
  values = similar(V,n...)
  TTArray(values,get_index_map(a))
end

function Base.sum(a::TTArray)
  sum(get_values(a))
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::TTArray,b::TTArray)
      @check get_index_map(a) == get_index_map(b)
      TTArray(($op)(get_values(a),get_values(b)),get_index_map(a))
    end
  end
end

(Base.:-)(a::TTArray) = a .* -1

function Base.:*(a::TTArray,b::Number)
  TTArray(get_values(a)*b,get_index_map(a))
end

function Base.:*(a::Number,b::TTArray)
  b*a
end

function Base.:/(a::TTArray,b::Number)
  a*(1/b)
end

function Base.:*(a::TTMatrix,b::TTVector)
  @check get_index_map(a) == get_index_map(b)
  TTArray(get_values(a)*get_values(b),get_index_map(a))
end

function Base.:\(a::TTMatrix,b::TTVector)
  @check get_index_map(a) == get_index_map(b)
  TTArray(get_values(a)\get_values(b),get_index_map(a))
end

function Base.transpose(a::TTArray)
  TTArray(transpose(get_values(a)),get_index_map(a))
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
  I::AbstractVector,J::AbstractVector,V::TTVector,m::Integer,n::Integer)
  TTArray(sparse(I,J,get_values(V),m,n),get_index_map(a))
end

SparseArrays.nnz(a::TTSparseMatrix) = nnz(get_values(a))
SparseArrays.nzrange(a::TTSparseMatrix,col::Int) = nzrange(get_values(a),col)
SparseArrays.rowvals(a::TTSparseMatrix) = rowvals(get_values(a))
SparseArrays.nonzeros(a::TTSparseMatrix) = TTArray(nonzeros(get_values(a)),get_index_map(a))
SparseMatricesCSR.colvals(a::TTSparseMatrix) = colvals(get_values(a))
SparseMatricesCSR.getoffset(a::TTSparseMatrix) = getoffset(get_values(a))

function SparseArrays.findnz(a::TTSparseMatrix)
  i,j,v = findnz(get_values(a))
  return i,j,TTArray(v,get_index_map(a))
end

LinearAlgebra.cholesky(a::TTSparseMatrix) = cholesky(get_values(a))

function Arrays.CachedArray(a::TTArray)
  TTArray(CachedArray(get_values(a)),get_index_map(a))
end

function Arrays.setsize!(
  a::TTArray{T,N,AbstractVector{CachedArray{T,N}}},
  s::NTuple{N,Int}) where {T,N}

  for ai in a
    setsize!(ai,s)
  end
  return a
end

struct TTBroadcast{V,M}
  values::V
  index_map::M
end

get_values(a::TTBroadcast) = a.values
get_index_map(a::TTBroadcast) = a.index_map

function Base.broadcasted(f,a::Union{TTArray,TTBroadcast}...)
  index_map = get_index_map(a[1])
  @check all(get_index_map.(a) == index_map)
  TTBroadcast(Base.broadcasted(f,map(get_values,a)...),index_map)
end

function Base.broadcasted(f,a::Union{TTArray,TTBroadcast},b::Number)
  TTBroadcast(Base.broadcasted(f,get_values(a),b),get_index_map(a))
end

function Base.broadcasted(f,a::Number,b::Union{TTArray,TTBroadcast})
  TTBroadcast(Base.broadcasted(f,a,get_values(b)),get_index_map(a))
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

function Base.materialize(b::TTBroadcast)
  a = Base.materialize(get_values(b))
  TTArray(a,get_values(a))
end

function Base.materialize!(a::TTArray,b::Broadcast.Broadcasted)
  Base.materialize!(get_values(a),b)
  a
end

function Base.materialize!(a::TTArray,b::TTBroadcast)
  Base.materialize!(get_values(a),get_values(b))
  a
end

Base.similar(a::TTBroadcast,args...) = TTBroadcast(similar(get_values(a),args...),get_index_map(a))
Base.axes(a::TTBroadcast,i...) = axes(get_values(a),i...)
Base.eachindex(a::TTBroadcast) = eachindex(get_values(a))
Base.IndexStyle(a::TTBroadcast) = IndexStyle(get_values(a))
Base.ndims(a::TTBroadcast) = ndims(get_values(a))
Base.ndims(::Type{TTBroadcast{D,V}}) where {D,V} = ndims(V)
Base.size(a::TTBroadcast) = size(get_values(a))
Base.length(a::TTBroadcast) = length(get_values(a))

# algebra

function Algebra.allocate_vector(::Type{V},index_map::AbstractIndexMap) where V<:TTArray
  values = Vector{eltype(V)}(undef,length(index_map))
  TTArray(values,index_map)
end
