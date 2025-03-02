const ParamReindex = Reindex{<:AbstractParamArray}

param_length(k::ParamReindex) = param_length(k.values)
param_getindex(k::ParamReindex,i::Integer) = Reindex(param_getindex(k.values,i))
MemoryLayoutStyle(k::ParamReindex) = MemoryLayoutStyle(k.values)

Arrays.testitem(k::ParamReindex) = param_getindex(k,1)
Arrays.testargs(k::ParamReindex,i...) = testargs(testitem(k),i...)

const PosNegParamReindex = PosNegReindex{<:AbstractParamArray,<:AbstractParamArray}

function param_length(k::PosNegParamReindex)
  @check param_length(k.values_pos) == param_length(k.values_neg)
  param_length(k.values_pos)
end

param_getindex(k::PosNegParamReindex,i::Integer) = PosNegReindex(param_getindex(k.values_pos,i),param_getindex(k.values_neg,i))

function MemoryLayoutStyle(k::PosNegParamReindex)
  @check MemoryLayoutStyle(k.values_pos)==MemoryLayoutStyle(k.values_neg)
  MemoryLayoutStyle(k.values_pos)
end

Arrays.testitem(k::PosNegParamReindex) = param_getindex(k,1)
Arrays.testargs(k::PosNegParamReindex,i...) = testargs(testitem(k),i...)

for T in (:ParamReindex,:PosNegParamReindex)
  @eval begin
    function Arrays.return_value(
      f::Broadcasting{<:$T},
      x::Union{Number,AbstractArray{<:Number}}...)

      vi = return_value(Broadcasting(testitem(f.f)),x...)
      parameterize(vi,param_length(f.f);style=MemoryLayoutStyle(f.f))
    end

    function Arrays.return_cache(
      f::Broadcasting{<:$T},
      x::Union{Number,AbstractArray{<:Number}}...)

      c = return_cache(Broadcasting(testitem(f.f)),x...)
      a = evaluate!(c,Broadcasting(testitem(f.f)),x...)
      cache = Vector{typeof(c)}(undef,param_length(f.f))
      data = parameterize(a,param_length(f.f);style=MemoryLayoutStyle(f.f))
      @inbounds for i = param_eachindex(f.f)
        cache[i] = return_cache(Broadcasting(param_getindex(f.f,i)),x...)
      end
      cache,data
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{<:$T},
      x::Union{Number,AbstractArray{<:Number}}...)

      c,data = cache
      @inbounds for i = param_eachindex(f.f)
        vi = evaluate!(c[i],Broadcasting(param_getindex(f.f,i)),x...)
        param_setindex!(data,vi,i)
      end
      data
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{<:$T},
      x::AbstractArray{<:Number})

      c,data = cache
      @inbounds for i = param_eachindex(f.f)
        vi = evaluate!(c[i],Broadcasting(param_getindex(f.f,i)),x)
        param_setindex!(data,vi,i)
      end
      data
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{<:$T},
      x::Number...)

      c,data = cache
      @inbounds for i = param_eachindex(f.f)
        vi = evaluate!(c[i],Broadcasting(param_getindex(f.f,i)),x...)
        param_setindex!(data,vi,i)
      end
      data
    end

    function Arrays.return_value(k::$T,j::Integer)
      vi = return_value(testitem(k),j)
      parameterize(vi,param_length(k);style=MemoryLayoutStyle(k))
    end

    function Arrays.return_cache(k::$T,j::Integer)
      c = return_cache(testitem(k),j)
      a = evaluate!(c,testitem(k),j)
      cache = Vector{typeof(c)}(undef,param_length(k))
      data = parameterize(a,param_length(k);style=MemoryLayoutStyle(k))
      for i = param_eachindex(k)
        cache[i] = return_cache(param_getindex(k,i),j)
      end
      cache,data
    end

    function Arrays.evaluate!(cache,k::$T,j::Integer)
      c,data = cache
      @inbounds for i = param_eachindex(k)
        vi = evaluate!(c[i],param_getindex(k,i),j)
        param_setindex!(data,vi,i)
      end
      data
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
    vi = evaluate!(c[i],k,param_getindex(values,i))
    param_setindex!(data,vi,i)
  end
  data
end
