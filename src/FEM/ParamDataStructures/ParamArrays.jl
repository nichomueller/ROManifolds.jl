# generic constructors

ParamArray(A::AbstractArray{<:Number},plength::Integer) = TrivialParamArray(A,plength)
ParamArray(a::AbstractArray{<:AbstractArray}) = ConsecutiveParamArray(a)

get_all_data(A::ParamArray) = @abstractmethod

function Base.setindex!(
  A::ParamArray{T,N},
  v::ParamArray{T′,N},
  i::Vararg{Integer,N}
  ) where {T,T′,N}

  setindex!(get_all_data(A),get_all_data(v),i...,:)
end

for f in (:(Base.maximum),:(Base.minimum))
  @eval begin
    $f(A::ParamArray) = $f(get_all_data(A))
    $f(g,A::ParamArray) = $f(g,get_all_data(A))
  end
end

for op in (:+,:-,:*,:/)
  @eval begin
    function ($op)(A::ParamArray,b::AbstractArray{<:Number})
      @check param_length(A) == length(b)
      B = copy(A)
      @inbounds for i in eachindex(B)
        bi = B[i]
        B[i] = $op(bi,fill(b[i],size(bi)))
      end
      return B
    end
  end
end

function (+)(b::AbstractArray{<:Number},A::ParamArray)
  (+)(A,b)
end

function (-)(b::AbstractArray{<:Number},A::ParamArray)
  (-)((-)(A,b))
end

function (*)(b::Number,A::ParamArray)
  (*)(A,b)
end

function Base.fill!(A::ParamArray,z::Number)
  fill!(get_all_data(A),z)
  return A
end

function LinearAlgebra.rmul!(A::ParamArray,b::Number)
  rmul!(get_all_data(A),b)
  return A
end

function LinearAlgebra.axpy!(α::Number,A::ParamArray,B::ParamArray)
  axpy!(α,get_all_data(A),get_all_data(B))
  return B
end

function Arrays.setsize!(A::ParamArray{T,N},s::NTuple{N,Integer}) where {T,N}
  setsize!(get_all_data(A),(s...,param_length(A)))
  A
end

function Base.getproperty(A::ParamArray,sym::Symbol)
  if sym == :array
    getfield(get_all_data(A),sym)
  else
    getfield(A,sym)
  end
end

"""
    struct TrivialParamArray{T,N,P<:AbstractArray{T,N}} <: ParamArray{T,N} end

Wrapper for nonparametric arrays that we wish assumed a parametric length.

"""
struct TrivialParamArray{T<:Number,N,A<:AbstractArray{T,N}} <: ParamArray{T,N}
  data::A
  plength::Int
  function TrivialParamArray(data::AbstractArray{T,N},plength::Int=1) where {T<:Number,N}
    A = typeof(data)
    new{T,N,A}(data,plength)
  end
end

param_length(A::TrivialParamArray) = A.plength
get_all_data(A::TrivialParamArray) = A.data

function TrivialParamArray(A::AbstractParamArray,args...)
  A
end

Base.size(A::TrivialParamArray{T,N}) where {T,N} = tfill(A.plength,Val{N}())

ArraysOfArrays.innersize(A::TrivialParamArray) = size(A.data)

Base.@propagate_inbounds function Base.getindex(A::TrivialParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    A.data
  else
    fill(zero(T),innersize(A))
  end
end

Base.@propagate_inbounds function Base.setindex!(A::TrivialParamArray,v,i::Integer...)
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  all(i.==iblock) && copyto!(A.data,v)
end

function Base.copy(A::TrivialParamArray)
  data′ = copy(get_all_data(A))
  TrivialParamArray(data′,param_length(A))
end

function Base.similar(A::TrivialParamArray{T,N},::Type{<:AbstractArray{T′}}) where {T,T′,N}
  data′ = similar(get_all_data(A),T′)
  TrivialParamArray(data′,param_length(A))
end

function Base.similar(A::TrivialParamArray{T,N},::Type{<:AbstractArray{T′}},dims::Dims{N}) where {T,T′,N}
  data′ = similar(get_all_data(A),T′,dims...)
  TrivialParamArray(data′,param_length(A))
end

function Base.copyto!(A::TrivialParamArray,B::TrivialParamArray)
  @check size(A) == size(B)
  copyto!(get_all_data(A),get_all_data(B))
  A
end

function Base.vec(A::TrivialParamArray)
  data′ = vec(get_all_data(A))
  TrivialParamArray(data′,param_length(A))
end

function Arrays.CachedArray(A::TrivialParamArray)
  data′ = CachedArray(get_all_data(A))
  TrivialParamArray(data′,param_length(A))
end

function get_param_entry(A::TrivialParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  entry = getindex(get_all_data(A),i...)
  fill(entry,param_length(A))
end

struct ConsecutiveParamArray{T,N,M,A<:AbstractArray{T,M}} <: ParamArray{T,N}
  data::A
  function ConsecutiveParamArray(data::AbstractArray{T,M}) where {T<:Number,M}
    N = M - 1
    A = typeof(data)
    new{T,N,M,A}(data)
  end
end

const ConsecutiveParamVector{T} = ConsecutiveParamArray{T,1,2,<:AbstractArray{T,2}}
const ConsecutiveParamMatrix{T} = ConsecutiveParamArray{T,2,3,<:AbstractArray{T,3}}

param_length(A::ConsecutiveParamArray{T,N,M}) where {T,N,M} = size(A.data,M)
get_all_data(A::ConsecutiveParamArray) = A.data

function ConsecutiveParamArray(a::AbstractVector{<:AbstractArray{T,N}}) where {T,N}
  if all(size(ai)==size(first(a)) for ai in a)
    ConsecutiveParamArray(stack(a))
  else
    if N==1
      GenericParamVector(a)
    elseif N==2
      GenericParamMatrix(a)
    else
      @notimplemented
    end
  end
end

function param_array(a::AbstractArray{<:Number,N},l::Integer) where N
  outer = (tfill(1,Val{N}())...,l)
  data = repeat(a;outer)
  ConsecutiveParamArray(data)
end

Base.size(A::ConsecutiveParamArray{T,N}) where {T,N} = tfill(param_length(A),Val{N}())

ArraysOfArrays.innersize(A::ConsecutiveParamArray{T,N}) where {T,N} = ArraysOfArrays.front_tuple(size(get_all_data(A)),Val{N}())

function Base.getindex(A::ConsecutiveParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    A.data[ArraysOfArrays._ncolons(Val{N}())...,iblock]
  else
    fill(zero(T),innersize(A))
  end
end

function Base.setindex!(A::ConsecutiveParamArray{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    A.data[ArraysOfArrays._ncolons(Val{N}())...,iblock] = v
  end
end

function Base.copy(A::ConsecutiveParamArray)
  data′ = copy(get_all_data(A))
  ConsecutiveParamArray(data′)
end

function Base.similar(A::ConsecutiveParamArray{T,N},::Type{<:AbstractArray{T′,N}}) where {T,T′,N}
  data′ = similar(get_all_data(A),T′)
  ConsecutiveParamArray(data′)
end

function Base.similar(A::ConsecutiveParamArray{T,N},::Type{<:AbstractArray{T′}},dims::Dims{N}) where {T,T′,N}
  pdims = (dims...,param_length(A))
  data′ = similar(get_all_data(A),T′,pdims...)
  ConsecutiveParamArray(data′)
end

function Base.copyto!(A::ConsecutiveParamArray,B::ConsecutiveParamArray)
  copyto!(get_all_data(A),get_all_data(B))
  A
end

function Base.vec(A::ConsecutiveParamArray)
  data′ = reshape(get_all_data(A),:,param_length(A))
  ConsecutiveParamArray(data′)
end

for op in (:+,:-)
  @eval begin
    function ($op)(A::ConsecutiveParamArray,B::ConsecutiveParamArray)
      AB = ($op)(get_all_data(A),get_all_data(B))
      ConsecutiveParamArray(AB)
    end
  end
end

for op in (:*,:/)
  @eval begin
    function ($op)(A::ConsecutiveParamArray,b::Number)
      Ab = ($op)(get_all_data(A),b)
      ConsecutiveParamArray(Ab)
    end
  end
end

function Arrays.CachedArray(A::ConsecutiveParamArray)
  data′ = CachedArray(get_all_data(A))
  ConsecutiveParamArray(data′)
end

function param_getindex(A::ConsecutiveParamArray{T,N},i::Integer) where {T,N}
  view(A.data,ArraysOfArrays._ncolons(Val{N}())...,i)
end

function get_param_entry(A::ConsecutiveParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  getindex(get_all_data(A),i...,:)
end

function get_param_entry(A::ConsecutiveParamArray,i...)
  data′ = view(get_all_data(A),i...,:)
  ConsecutiveParamArray(data′)
end

"""
    struct ParamArray{T,N,P<:AbstractVector{<:AbstractArray{T,N}}} <: ParamArray{T,N} end

Represents conceptually a vector of arrays, but the entries are stored in
consecutive memory addresses. So in practice it simply wraps an AbstractArray,
with a parametric length equal to its last dimension

"""
struct GenericParamVector{Tv,Ti} <: ParamArray{Tv,1}
  data::Vector{Tv}
  ptrs::Vector{Ti}
end

param_length(A::GenericParamVector) = length(A.ptrs)-1
get_all_data(A::GenericParamVector) = A.data
get_ptrs(A::GenericParamVector) = A.ptrs

function GenericParamVector(a::AbstractVector{<:AbstractVector{Tv}}) where Tv
  ptrs = _vec_of_pointers(a)
  n = length(a)
  u = one(eltype(ptrs))
  ndata = ptrs[end]-u
  data = Vector{Tv}(undef,ndata)
  p = 1
  @inbounds for i in 1:n
    ai = a[i]
    for j in 1:length(ai)
      aij = ai[j]
      data[p] = aij
      p += 1
    end
  end
  GenericParamVector(data,ptrs)
end

Base.size(A::GenericParamVector) = (length(A.ptrs)-1,)

function ArraysOfArrays.innersize(A::GenericParamVector)
  innerlength = ptrs[2]-ptrs[1]
  @check all(ptrs[i+1]-ptrs[i] == innerlength for i in 1:length(ptrs)-1)
  (innerlength,)
end

function Base.getindex(A::GenericParamVector,i::Integer)
  @boundscheck checkbounds(A,i)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  A.data[pini:pend]
end

function Base.setindex!(A::GenericParamVector,v,i::Integer)
  @boundscheck checkbounds(A,i)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  A.data[pini:pend] = v
end

function Base.copy(A::GenericParamVector)
  data′ = copy(get_all_data(A))
  ptrs = get_ptrs(A)
  GenericParamVector(data′,ptrs)
end

function Base.similar(A::GenericParamVector{T},::Type{<:AbstractVector{T′}}) where {T,T′}
  data′ = similar(get_all_data(A),T′)
  ptrs = get_ptrs(A)
  GenericParamVector(data′,ptrs)
end

function Base.copyto!(A::GenericParamVector,B::GenericParamVector)
  @check get_ptrs(A) == get_ptrs(B)
  copyto!(get_all_data(A),get_all_data(B))
  A
end

function Base.vec(A::GenericParamVector)
  A
end

for op in (:+,:-)
  @eval begin
    function ($op)(A::GenericParamVector,B::GenericParamVector)
      @check get_ptrs(A) == get_ptrs(B)
      AB = ($op)(get_all_data(A),get_all_data(B))
      ptrs = get_ptrs(A)
      GenericParamVector(AB,ptrs)
    end
  end
end

for op in (:*,:/)
  @eval begin
    function ($op)(A::GenericParamVector,b::Number)
      Ab = ($op)(get_all_data(A),b)
      ptrs = get_ptrs(A)
      GenericParamVector(AB,ptrs)
    end
  end
end

for op in (:+,:-,:*,:/)
  @eval begin
    function ($op)(A::GenericParamVector,b::AbstractArray{<:Number})
      @check param_length(A) == length(b)
      B = copy(A)
      @inbounds for i in eachindex(B)
        bi = B[i]
        B[i] = $op(bi,b[i])
      end
      return B
    end
  end
end

function Arrays.CachedArray(A::GenericParamVector)
  data′ = CachedArray(get_all_data(A))
  ptrs = get_ptrs(A)
  GenericParamVector(data′,ptrs)
end

function get_param_entry(A::GenericParamVector{T},i::Integer) where T
  entries = Vector{T}(undef,param_length(A))
  @inbounds for k in param_eachindex(A)
    entries[k] = A[k][i]
  end
  entries
end

struct GenericParamMatrix{Tv,Ti} <: ParamArray{Tv,2}
  data::Vector{Tv}
  ptrs::Vector{Ti}
  nrows::Vector{Ti}
end

param_length(A::GenericParamMatrix) = length(A.ptrs)-1
get_all_data(A::GenericParamMatrix) = A.data
get_ptrs(A::GenericParamMatrix) = A.data
get_nrows(A::GenericParamMatrix) = A.nrows

function GenericParamMatrix(a::AbstractVector{<:AbstractMatrix{Tv}}) where Tv
  ptrs = _vec_of_pointers(a)
  n = length(a)
  Ti = eltype(ptrs)
  u = one(Ti)
  ndata = ptrs[end]-u
  data = Vector{Tv}(undef,ndata)
  nrows = Vector{Ti}(undef,n)
  p = 1
  @inbounds for i in 1:n
    ai = a[i]
    nrows[i] = size(ai,1)
    for j in 1:length(ai)
      aij = ai[j]
      data[p] = aij
      p += 1
    end
  end
  GenericParamMatrix(data,ptrs,nrows)
end

Base.size(A::GenericParamMatrix) = (length(A.ptrs)-1,length(A.ptrs)-1)

ArraysOfArrays.innersize(A::GenericParamMatrix) = @notimplemented

function Base.getindex(A::GenericParamMatrix{Tv},i::Integer,j::Integer) where Tv
  @boundscheck checkbounds(A,i,j)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  nrow = A.nrows[i]
  ncol = Int((pend-pini+1)/nrow)
  if i == j
    reshape(A.data[pini:pend],nrow,ncol)
  else
    fill(zero(Tv),nrow,ncol)
  end
end

function Base.setindex!(A::GenericParamMatrix,v,i::Integer,j::Integer)
  @boundscheck checkbounds(A,i,j)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  if i == j
    A.data[pini:pend] = vec(v)
  end
end

function Base.copy(A::GenericParamMatrix)
  data′ = copy(get_all_data(A))
  ptrs = get_ptrs(A)
  nrows = get_nrows(A)
  GenericParamMatrix(data′,ptrs,nrows)
end

function Base.similar(A::GenericParamMatrix{T,N},::Type{<:AbstractMatrix{T′}}) where {T,T′,N}
  data′ = similar(get_all_data(A),T′)
  ptrs = get_ptrs(A)
  nrows = get_nrows(A)
  GenericParamMatrix(data′,ptrs,nrows)
end

function Base.copyto!(A::GenericParamMatrix,B::GenericParamMatrix)
  @check get_ptrs(A) == get_ptrs(B)
  @check get_nrows(A) == get_nrows(B)
  copyto!(get_all_data(A),get_all_data(B))
  A
end

function Base.vec(A::GenericParamMatrix)
  data = get_all_data(A)
  ptrs = get_ptrs(A)
  GenericParamVector(data,ptrs)
end

for op in (:+,:-)
  @eval begin
    function ($op)(A::GenericParamMatrix,B::GenericParamMatrix)
      @check get_ptrs(A) == get_ptrs(B)
      @check get_nrows(A) == get_nrows(B)
      AB = ($op)(get_all_data(A),get_all_data(B))
      GenericParamMatrix(AB,get_ptrs(A),get_nrows(A))
    end
  end
end

for op in (:*,:/)
  @eval begin
    function ($op)(A::GenericParamMatrix,b::Number)
      Ab = ($op)(get_all_data(A),b)
      GenericParamMatrix(AB,get_ptrs(A),get_nrows(A))
    end
  end
end

function Arrays.CachedArray(A::GenericParamMatrix)
  data′ = CachedArray(get_all_data(A))
  ptrs = get_ptrs(A)
  nrows = get_nrows(A)
  GenericParamMatrix(data′,ptrs,nrows)
end

function get_param_entry(A::GenericParamMatrix{T},i::Integer,j::Integer) where T
  entries = Vector{T}(undef,param_length(A))
  @inbounds for k in param_eachindex(A)
    entries[k] = A[k][i,j]
  end
  entries
end

# utils

function _vec_of_pointers(a::AbstractVector{<:AbstractArray})
  n = length(a)
  ptrs = Vector{Int}(undef,n+1)
  @inbounds for i in 1:n
    ai = a[i]
    ptrs[i+1] = length(ai)
  end
  length_to_ptrs!(ptrs)
  ptrs
end
