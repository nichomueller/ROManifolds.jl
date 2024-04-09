abstract type AbstractParamContainer{T,N} <: AbstractArray{T,N} end
abstract type AbstractParamBroadcast end

struct ParamContainer{T,A,L} <: AbstractParamContainer{T,1}
  array::A
  function ParamContainer(array::AbstractVector{T},::Val{L}) where {T,L}
    A = typeof(array)
    new{T,A,L}(array)
  end
end

ParamContainer(array::AbstractVector{T}) where T = ParamContainer(array,Val(length(array)))
ParamContainer(array::AbstractVector{T}) where T<:AbstractArray = ParamArray(array,Val(length(array)))

Arrays.get_array(a::ParamContainer) = a.array
Arrays.testitem(c::ParamContainer) = testitem(get_array(c))
Base.length(c::ParamContainer{T,A,L}) where {T,A,L} = L
Base.size(c::ParamContainer) = (length(c),)
Base.eachindex(c::ParamContainer) = Base.OneTo(length(c))
Base.getindex(c::ParamContainer,i...) = getindex(get_array(c),i...)
Base.setindex!(c::ParamContainer,v,i...) = setindex!(get_array(c),v,i...)

function Base.transpose(a::ParamContainer)
  at = map(transpose,get_array(a))
  ParamContainer(at)
end

function Base.:+(a::T,b::T) where T<:ParamContainer
  c = similar(a.array)
  @inbounds for i = eachindex(a)
    c[i] = a[i] + b[i]
  end
  ParamContainer(c)
end

function Base.:-(a::T,b::T) where T<:ParamContainer
  c = similar(a.array)
  @inbounds for i = eachindex(a)
    c[i] = a[i] - b[i]
  end
  ParamContainer(c)
end

for T in (:(Point),:(AbstractVector{<:Point}))
  @eval begin
    function Arrays.return_cache(f::ParamContainer,x::$T)
      ci = return_cache(testitem(f),x)
      ai = evaluate!(ci,testitem(f),x)
      cache = Vector{typeof(ci)}(undef,length(f))
      array = Vector{typeof(ai)}(undef,length(f))
      for i = eachindex(f)
        cache[i] = return_cache(f[i],x)
      end
      cache,ParamContainer(array)
    end

    function Arrays.evaluate!(cache,f::ParamContainer,x::$T)
      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f[i],x)
      end
      array
    end
  end
end



# function Arrays.return_value(
#   k::Broadcasting{<:Operation},
#   args::Union{Field,ParamContainer{<:Field,<:Any,L}}...) where L

#   v = return_value(k,_get_field(args,1)...)
#   array = Vector{typeof(v)}(undef,L)
#   for i = eachindex(array)
#     array[i] = return_value(k,_get_field(args,i)...)
#   end
#   ParamContainer(array)
# end

# function Arrays.evaluate!(
#   cache,
#   k::Broadcasting{<:Operation},
#   args::Union{Field,ParamContainer{<:Field,<:Any,L}}...) where L

#   @check isnothing(cache)
#   v = evaluate!(cache,k,_get_field(args,1)...)
#   array = Vector{typeof(v)}(undef,L)
#   for i = eachindex(array)
#     array[i] = evaluate!(cache,k,_get_field(args,i)...)
#   end
#   ParamContainer(array)
# end

# function Arrays.return_value(k::Broadcasting{<:Operation},args::Union{Field,ParamContainer{<:Field}}...)
#   BroadcastOpParamFieldArray(k.f.op,args...)
# end

# function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},args::Union{Field,ParamContainer{<:Field}}...)
#   BroadcastOpParamFieldArray(k.f.op,args...)
# end

# struct BroadcastOpParamFieldArray{O,T,L,A} <: AbstractParamContainer{T,1}
#   op::O
#   args::A
#   function BroadcastOpParamFieldArray(op,args::Union{Field,AbstractParamContainer{<:Field}}...)
#     fs = map(testitem,args)
#     T = return_type(Operation(op),fs...)
#     L = length()
#     A = typeof(args)
#     O = typeof(op)
#     new{O,T,L,A}(op,args)
#   end
# end

# function Fields.BroadcastOpFieldArray(op,args::Union{Field,AbstractParamContainer{<:Field}}...)
#   BroadcastOpParamFieldArray(op,args...)
# end

# Base.length(a::BroadcastOpParamFieldArray{O,T,L,A}) where {O,T,L,A} = L
# Base.size(a::BroadcastOpParamFieldArray) = (length(a),)

# function Base.getindex(a::BroadcastOpParamFieldArray,i::Integer)
#   _get_field(a::Field,i) = a
#   _get_field(a::ParamContainer{<:Field},i) = a[i]
#   argi = map(x->_get_field(x,i),a.args)
#   Operation(a.op)(argi...)
# end

# function Arrays.testitem(a::BroadcastOpParamFieldArray)
#   fs = map(testitem,a.args)
#   return_value(Operation(a.op),fs...)
# end

# for T in (:(Point),:(AbstractArray{<:Point}))
#   @eval begin

#     function Arrays.return_cache(f::BroadcastOpParamFieldArray,x::$T)
#       c = return_cache(testitem(f),x)
#       cache = Vector{typeof(c)}(undef,length(f))
#       for i = eachindex(cache)
#         cache[i] = return_cache(f[i],x)
#       end
#       c,ParamContainer(cache)
#     end

#     function Arrays.evaluate!(cache,f::BroadcastOpParamFieldArray,x::$T)
#       c,pcache = cache
#       @inbounds for i = eachindex(pcache)
#         pcache[i] = evaluate!(c,f[i],x)
#       end
#       return pcache
#     end

#   end
# end
