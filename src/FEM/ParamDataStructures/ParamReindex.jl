const ParamReindex = Reindex{<:AbstractParamArray}

param_length(k::ParamReindex) = param_length(k.values)
param_getindex(k::ParamReindex,i::Integer) = Reindex(param_getindex(k.values,i))

Arrays.testitem(k::ParamReindex) = param_getindex(k,1)
Arrays.testargs(k::ParamReindex,i...) = testargs(testitem(k),i...)

const PosNegParamReindex = PosNegReindex{<:AbstractParamArray,<:AbstractParamArray}

function param_length(k::PosNegParamReindex)
  @check param_length(k.values_pos) == param_length(k.values_neg)
  param_length(k.values_pos)
end

param_getindex(k::PosNegParamReindex,i::Integer) = PosNegReindex(param_getindex(k.values_pos,i),param_getindex(k.values_neg,i))

Arrays.testitem(k::PosNegParamReindex) = param_getindex(k,1)
Arrays.testargs(k::PosNegParamReindex,i...) = testargs(testitem(k),i...)

for T in (:ParamReindex,:PosNegParamReindex)
  @eval begin
    function Arrays.return_value(
      f::Broadcasting{<:$T},
      x::Union{Number,AbstractArray{<:Number}}...)

      v = return_value(Broadcasting(testitem(f.f)),x...)
      array = Vector{typeof(v)}(undef,param_length(f.f))
      @inbounds for i = param_eachindex(f.f)
        array[i] = return_value(Broadcasting(param_getindex(f.f,i)),x...)
      end
      ParamArray(array)
    end

    function Arrays.return_cache(
      f::Broadcasting{<:$T},
      x::Union{Number,AbstractArray{<:Number}}...)

      c = return_cache(Broadcasting(testitem(f.f)),x...)
      a = evaluate!(c,Broadcasting(testitem(f.f)),x...)
      cache = Vector{typeof(c)}(undef,param_length(f.f))
      data = parameterize(a,param_length(f.f))
      @inbounds for i = param_eachindex(f.f)
        cache[i] = return_cache(Broadcasting(param_getindex(f.f,i)),x...)
      end
      cache,data
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{<:$T},
      x::Union{Number,AbstractArray{<:Number}}...)

      cx,array = cache
      @inbounds for i = param_eachindex(f.f)
        array[i] = evaluate!(cx[i],Broadcasting(param_getindex(f.f,i)),x...)
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{<:$T},
      x::AbstractArray{<:Number})

      cx,array = cache
      @inbounds for i = param_eachindex(f.f)
        array[i] = evaluate!(cx[i],Broadcasting(param_getindex(f.f,i)),x)
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{<:$T},
      x::Number...)

      cx,array = cache
      @inbounds for i = param_eachindex(f.f)
        array[i] = evaluate!(cx[i],Broadcasting(param_getindex(f.f,i)),x...)
      end
      array
    end

    function Arrays.return_value(k::$T,j::Integer)
      v = return_value(testitem(k),j)
      array = Vector{typeof(v)}(undef,param_length(k))
      @inbounds for i = param_eachindex(k)
        array[i] = return_value(param_getindex(k,i),j)
      end
      ParamArray(array)
    end

    function Arrays.return_cache(k::$T,j::Integer)
      c = return_cache(testitem(k),j)
      a = evaluate!(c,testitem(k),j)
      cache = Vector{typeof(c)}(undef,param_length(k))
      data = parameterize(a,param_length(k))
      for i = param_eachindex(k)
        cache[i] = return_cache(param_getindex(k,i),j)
      end
      cache,data
    end

    function Arrays.evaluate!(cache,k::$T,j::Integer)
      cx,array = cache
      @inbounds for i = param_eachindex(k)
        array[i] = evaluate!(cx[i],param_getindex(k,i),j)
      end
      array
    end
  end
end

function Arrays.return_cache(k::OReindex,values::AbstractParamVector)
  v = testitem(values)
  c = return_cache(k,v)
  a = evaluate!(c,k,v)
  data = parameterize(a,param_length(values);style=MemoryLayoutStyle(values))
  cache = Vector{typeof(c)}(undef,param_length(values))
  for i = param_eachindex(values)
    cache[i] = return_cache(k,param_getindex(values,i))
  end
  cache,data
end

function Arrays.evaluate!(cache,k::OReindex,values::AbstractParamVector)
  c,data = cache
  @inbounds for i = param_eachindex(values)
    data[i] = evaluate!(c[i],k,param_getindex(values,i))
  end
  data
end
