const ParamReindex = Reindex{A} where A<:ParamArray

function Base.length(k::ParamReindex)
  length(k.values)
end

Arrays.testitem(k::ParamReindex) = Reindex(testitem(k.values))
Arrays.testargs(k::ParamReindex,i::Integer...) = testargs(testitem(k),i...)
Arrays.testargs(k::ParamReindex,i...) = testargs(testitem(k),i...)

function Base.getindex(k::ParamReindex,i::Integer)
  Reindex(k.values[i])
end

const PosNegParamReindex = PosNegReindex{A,B} where {A<:ParamArray,B<:ParamArray}

function Base.length(k::PosNegParamReindex)
  @assert length(k.values_pos) == length(k.values_neg)
  length(k.values_pos)
end

Arrays.testitem(k::PosNegParamReindex) = PosNegReindex(testitem(k.values_pos),testitem(k.values_neg))
Arrays.testargs(k::PosNegParamReindex,i::Integer) = testargs(testitem(k),i)

function Base.iterate(k::PosNegParamReindex,i::Integer)
  PosNegReindex(k.values_pos[i],k.values_neg[i])
end

function Arrays.return_value(
  f::Broadcasting{<:PosNegParamReindex},
  x::Union{Number,AbstractArray{<:Number}}...)

  v = return_value(Broadcasting(testitem(f.f)),x...)
  array = Vector{typeof(v)}(undef,length(f.f))
  for i = 1:length(f.f)
    array[i] = return_value(Broadcasting(f.f[i]),x...)
  end
  ParamArray(array)
end

function Arrays.return_cache(
  f::Broadcasting{<:PosNegParamReindex},
  x::Union{Number,AbstractArray{<:Number}}...)

  c = return_cache(Broadcasting(testitem(f.f)),x...)
  a = evaluate!(c,Broadcasting(testitem(f.f)),x...)
  cache = Vector{typeof(c)}(undef,length(f.f))
  array = Vector{typeof(a)}(undef,length(f.f))
  for i = 1:length(f.f)
    cache[i] = return_cache(Broadcasting(f.f[i]),x...)
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{<:PosNegParamReindex},
  x::Union{Number,AbstractArray{<:Number}}...)

  cx,array = cache
  @inbounds for i = 1:length(f.f)
    array[i] = evaluate!(cx[i],Broadcasting(f.f[i]),x...)
  end
  array
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{<:PosNegParamReindex},
  x::AbstractArray{<:Number})

  cx,array = cache
  @inbounds for i = 1:length(f.f)
    array[i] = evaluate!(cx[i],Broadcasting(f.f[i]),x)
  end
  array
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{<:PosNegParamReindex},
  x::Number...)

  cx,array = cache
  @inbounds for i = 1:length(f.f)
    array[i] = evaluate!(cx[i],Broadcasting(f.f[i]),x...)
  end
  array
end

for T in (:ParamReindex,:PosNegParamReindex)
  @eval begin
    function Arrays.return_value(k::$T,j::Integer)
      v = return_value(testitem(k),j)
      array = Vector{typeof(v)}(undef,length(k))
      for i = 1:length(k)
        array[i] = return_value(k[i],j)
      end
      ParamArray(array)
    end

    function Arrays.return_cache(k::$T,j::Integer)
      c = return_cache(testitem(k),j)
      a = evaluate!(c,testitem(k),j)
      cache = Vector{typeof(c)}(undef,length(k))
      array = Vector{typeof(a)}(undef,length(k))
      for i = 1:length(k)
        cache[i] = return_cache(k[i],j)
      end
      cache,ParamArray(array)
    end

    function Arrays.evaluate!(cache,k::$T,j::Integer)
      cx,array = cache
      @inbounds for i = 1:length(k)
        array[i] = evaluate!(cx[i],k[i],j)
      end
      array
    end

    function Arrays.evaluate(k::$T,j::Integer)
      cache = return_cache(k,j)
      array = evaluate!(cache,k,j)
      ParamArray(array)
    end
  end
end
