abstract type PTOrdering end
struct ParamOutTimeIn <: PTOrdering end
struct TimeOutParamIn <: PTOrdering end

struct PTArray{T}
  array::Vector{Vector{T}}
  axis::PTOrdering

  function PTArray(array::Vector{Vector{T}},axis=ParamOutTimeIn()) where T
    new{T}(array,axis)
  end
end

Base.size(b::PTArray) = size(b.array)
Base.length(b::PTArray) = length(b.array)
Base.eltype(::Type{<:PTArray{T}}) where T = T
Base.eltype(::PTArray{T}) where T = T
Base.ndims(b::PTArray) = 1
Base.ndims(::Type{PTArray}) = 1
function Base.getindex(b::PTArray,i...)
  b.array[i...]
end
function Base.setindex!(b::PTArray,v,i...)
  b.array[i...] = v
end
function Base.show(io::IO,o::PTArray)
  print(io,"PTArray($(o.array), $(o.axis))")
end

function Arrays.testitem(f::PTArray{T}) where T
  @notimplementedif !isconcretetype(T)
  if length(f) != 0
    f.array[1]
  else
    testvalue(T)
  end
end

function Arrays.testvalue(::Type{PTArray{T}}) where T
  s = ntuple(i->0,Val(1))
  array = Vector{Vector{T}}(undef,s)
  PTArray(array)
end

function Arrays.CachedArray(a::PTArray)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  for i in eachindex(a.array)
    array[i] = CachedArray(a.array[i])
  end
  PTArray(array,a.axis)
end

function Base.:≈(a::AbstractArray{<:PTArray},b::AbstractArray{<:PTArray})
  all(z->z[1]≈z[2],zip(a,b))
end

function Base.:≈(a::PTArray,b::PTArray)
  if size(a) != size(b) || a.axis != b.axis
    return false
  end
  for i in eachindex(a.array)
    if !(a.array[i] ≈ b.array[i])
      return false
    end
  end
  true
end

function Base.:(==)(a::PTArray,b::PTArray)
  if size(a) != size(b) || a.axis != b.axis
    return false
  end
  for i in eachindex(a.array)
    if !(a.array[i] == b.array[i])
      return false
    end
  end
  true
end

Base.copy(a::PTArray) = PTArray(copy(a.array),copy(a.axis))
Base.eachindex(a::PTArray) = eachindex(a.array)
