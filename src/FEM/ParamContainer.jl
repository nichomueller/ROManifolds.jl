abstract type AbstractParamContainer{T,N} <: AbstractArray{T,N} end

struct ParamContainer{T,L} <: AbstractParamContainer{T,1}
  array::AbstractVector{T}
  function ParamContainer(array::AbstractVector{T},::Val{L}) where {T,L}
    new{T,L}(array)
  end
end

ParamContainer(array::AbstractVector{T}) where T = ParamContainer(array,Val(length(array)))
ParamContainer(array::AbstractVector{T}) where T<:AbstractArray = ParamArray(array,Val(length(array)))

Base.length(c::ParamContainer{T,L}) where {T,L} = L
Base.size(c::ParamContainer) = (length(c),)
Base.eachindex(c::ParamContainer) = Base.OneTo(length(c))
Base.getindex(c::ParamContainer,i::Integer) = c.array[i]
Base.iterate(c::ParamContainer,i...) = iterate(c.array,i...)
Arrays.testitem(c::ParamContainer) = testitem(c.array)

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
