"""
    abstract type ParamArray{T,N,L} <: AbstractParamArray{T,N,L,Array{T,N}} end

Type representing parametric arrays of type A. L encodes the parametric length.
Subtypes:
- [`ParamArray`](@ref).
- [`ConsecutiveParamArray`](@ref).
- [`TrivialParamArray`](@ref).
- [`BlockParamArray`](@ref).

"""
abstract type ParamArray{T,N,L} <: AbstractParamArray{T,N,L,Array{T,N}} end
const ParamVector{T,L} = ParamArray{T,1,L}
const ParamMatrix{T,L} = ParamArray{T,2,L}

# generic constructors

ParamArray(A::AbstractArray{<:Number},plength::Integer) = TrivialParamArray(A,plength)
ParamArray(a::AbstractVector{<:AbstractArray}) = ConsecutiveParamArray(a)

function Base.setindex!(
  A::ParamArray{T,N,L},
  v::ParamArray{T′,N,L},
  i::Vararg{Integer}) where {T,T′,N,L}

  setindex!(get_all_data(A),get_all_data(v),i...,:)
end

get_all_data(A::ParamArray) = @abstractmethod

function (+)(b::AbstractArray{<:Number},A::ParamArray)
  (+)(A,b)
end

function (-)(b::AbstractArray{<:Number},A::ParamArray)
  (-)((-)(A,b))
end

for f in (:(Base.maximum),:(Base.minimum))
  @eval begin
    $f(A::ParamArray) = $f(get_all_data(A))
    $f(g,A::ParamArray) = $f(g,get_all_data(A))
  end
end

function Base.transpose(A::AbstractParamArray)
  @notimplemented "do I need this?"
end

function Base.vec(A::AbstractParamArray)
  @notimplemented "do I need this?"
end

for op in (:+,:-,:*,:/)
  @eval begin
    function ($op)(A::ParamArray,b::AbstractArray{<:Number})
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

function Base.fill!(A::ParamArray,z::Number)
  fill!(get_all_data(A),z)
  return A
end

function LinearAlgebra.rmul!(A::ParamArray,b::Number)
  rmul!(get_all_data(A),b)
  return A
end

function LinearAlgebra.axpy!(α::Number,A::ParamArray,B::ParamArray)
  @check size(A) == size(B)
  axpy!(α,get_all_data(A),get_all_data(B))
  return B
end

function (*)(A::ParamArray,B::ParamArray)
  @notimplemented
end

function param_view(A::ParamArray,i::Union{Integer,AbstractVector,Colon}...)
  @deprecate
  # ParamArray(view(get_all_data(A),i...,:))
end

function Arrays.setsize!(A::ParamArray{T,N},s::NTuple{N,Integer}) where {T,N}
  setsize!(get_all_data(A))
end

"""
    struct TrivialParamArray{T,N,L,P<:AbstractArray{T,N}} <: ParamArray{T,N,L} end

Wrapper for nonparametric arrays that we wish assumed a parametric length.

"""
struct TrivialParamArray{T<:Number,N,L} <: ParamArray{T,N,L}
  data::Array{T,N}
  plength::Int
  function TrivialParamArray(data::Array{T,N},plength::Int=1) where {T<:Number,N}
    new{T,N,plength}(data,plength)
  end
end

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

function Base.similar(A::TrivialParamArray{T,N},::Type{<:AbstractArray{T′}},dims::Dims{N}) where {T,T′,N,N}
  data′ = similar(get_all_data(A),T′,dims...)
  TrivialParamArray(data′,param_length(A))
end

function Base.copyto!(A::TrivialParamArray,B::TrivialParamArray)
  @check size(A) == size(B)
  copyto!(get_all_data(A),get_all_data(B))
  A
end

function Arrays.CachedArray(A::TrivialParamArray)
  data′ = CachedArray(get_all_data(A))
  TrivialParamArray(data′,param_length(A))
end

function get_param_entry(A::TrivialParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  entry = getindex(get_all_data(A),i...)
  fill(entry,param_length(A))
end

struct ConsecutiveParamArray{T,N,L,M} <: ParamArray{T,N,L}
  data::Array{T,M}
  function ConsecutiveParamArray(data::Array{T,M}) where {T,M}
    N = M - 1
    L = size(data,M)
    new{T,N,L,M}(data)
  end
end

const ConsecutiveParamVector{T,L} = ConsecutiveParamArray{T,1,L,2}
const ConsecutiveParamMatrix{T,L} = ConsecutiveParamArray{T,2,L,3}

get_all_data(A::ConsecutiveParamArray) = A.data

function ConsecutiveParamArray(a::AbstractVector{<:AbstractArray})
  if all(size(ai)==size(first(a)))
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

ArraysOfArrays.innersize(A::ConsecutiveParamArray) = ArraysOfArrays.front_tuple(get_all_data(A),Val{N}())

function Base.getindex(A::ConsecutiveParamArray{T},i::Integer,j::Integer) where T
  @boundscheck checkbounds(A,i,j)
  if i == j
    A.data[ArraysOfArrays._ncolons(Val{N}())...,i]
  else
    fill(zero(T),innersize(A))
  end
end

function Base.setindex!(A::ConsecutiveParamArray,v,i::Integer,j::Integer)
  @boundscheck checkbounds(A,i,j)
  if i == j
    A.data[ArraysOfArrays._ncolons(Val{N}())...,i] = v
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

function Base.similar(A::ConsecutiveParamArray{T,N},::Type{<:AbstractArray{T′}},dims::Dims{N}) where {T,T′,N,N}
  pdims = (dims...,param_length(A))
  data′ = similar(get_all_data(A),T′,pdims...)
  ConsecutiveParamArray(data′)
end

function Base.copyto!(A::ConsecutiveParamArray,B::ConsecutiveParamArray)
  copyto!(get_all_data(A),get_all_data(B))
  A
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
      ConsecutiveParamArray(AB)
    end
  end
end

function (*)(A::ConsecutiveParamArray,B::ConsecutiveParamArray)
  get_all_data(A)*get_all_data(B)
end
function (*)(A::ConsecutiveParamArray,B::AbstractArray)
  get_all_data(A)*B
end
function (*)(A::AbstractArray,B::ConsecutiveParamArray)
  A*get_all_data(B)
end

function Arrays.CachedArray(A::ConsecutiveParamArray)
  data′ = CachedArray(get_all_data(A))
  ConsecutiveParamArray(data′)
end

function get_param_entry(A::ConsecutiveParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  getindex(get_all_data(A),i...,:)
end

"""
    struct ParamArray{T,N,L,P<:AbstractVector{<:AbstractArray{T,N}}} <: ParamArray{T,N,L} end

Represents conceptually a vector of arrays, but the entries are stored in
consecutive memory addresses. So in practice it simply wraps an AbstractArray,
with a parametric length equal to its last dimension

"""
struct GenericParamVector{Tv,L,Ti} <: ParamArray{Tv,1,L}
  data::Vector{Tv}
  ptrs::Vector{Ti}
  function GenericParamVector(data::Vector{Tv},ptrs::Vector{Ti}) where {Tv,Ti}
    L = length(ptrs)-1
    new{Tv,L,Ti}(data,ptrs)
  end
end

get_all_data(A::GenericParamVector) = A.data
get_ptrs(A::GenericParamVector) = A.ptrs

function GenericParamVector(a::AbstractVector{<:AbstractVector{Tv}}) where Tv
  ptrs = _vec_of_pointers(a)
  n = length(a)
  u = one(Ti)
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

function get_param_entry(A::GenericParamVector{T,L},i::Integer) where {T,L}
  entries = Vector{T}(undef,L)
  @inbounds for k in param_eachindex(A)
    entries[k] = A[k][i]
  end
  entries
end

struct GenericParamMatrix{Tv,Ti} <: AbstractMatrix{Tv}
  data::Vector{Tv}
  ptrs::Vector{Ti}
  nrows::Vector{Ti}
end

get_all_data(A::GenericParamMatrix) = A.data
get_ptrs(A::GenericParamMatrix) = A.data
get_nrows(A::GenericParamMatrix) = A.nrows

function GenericParamMatrix(
  a::AbstractVector{<:AbstractMatrix{Tv}},
  ptrs::Vector{Ti},
  nrows::Vector{Ti}) where {Tv,Ti}

  n = length(a)
  u = one(Ti)
  ptrs = _vec_of_pointers(a)
  ndata = ptrs[end]-u
  data = Vector{Tv}(undef,ndata)
  nrows = Vector{Int}(undef,n)
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

function get_param_entry(A::GenericParamMatrix{T,L},i::Integer,j::Integer) where {T,L}
  entries = Vector{T}(undef,L)
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
