const ParamReindex = Reindex{A} where A<:ParamArray

function Base.length(k::ParamReindex)
  length(k.values)
end

Arrays.testitem(k::ParamReindex) = Reindex(testitem(k.values))
Arrays.testargs(k::ParamReindex,i::Integer...) = testargs(testitem(k),i...)
Arrays.testargs(k::ParamReindex,i...) = testargs(testitem(k),i...)

function Base.iterate(k::ParamReindex,oldstate...)
  it = iterate(k.values,oldstate...)
  if isnothing(it)
    return nothing
  end
  vit,nextstate = it
  Reindex(vit),nextstate
end

const PosNegParamReindex = PosNegReindex{A,B} where {A<:ParamArray,B<:ParamArray}

function Base.length(k::PosNegParamReindex)
  @assert length(k.values_pos) == length(k.values_neg)
  length(k.values_pos)
end

Arrays.testitem(k::PosNegParamReindex) = PosNegReindex(testitem(k.values_pos),testitem(k.values_neg))
Arrays.testargs(k::PosNegParamReindex,i::Integer) = testargs(testitem(k),i)

function Base.iterate(k::PosNegParamReindex)
  itpos = iterate(k.values_pos)
  itneg = iterate(k.values_neg)
  if isnothing(itpos) && isnothing(itneg)
    return nothing
  end
  vitpos,nextstatepos = itpos
  vitneg,nextstateneg = itneg
  nextstate = nextstatepos,nextstateneg
  PosNegReindex(vitpos,vitneg),nextstate
end

function Base.iterate(k::PosNegParamReindex,oldstate)
  oldstatepos,oldstatneg = oldstate
  itpos = iterate(k.values_pos,oldstatepos)
  itneg = iterate(k.values_neg,oldstatneg)
  if isnothing(itpos) && isnothing(itneg)
    return nothing
  end
  vitpos,nextstatepos = itpos
  vitneg,nextstateneg = itneg
  nextstate = nextstatepos,nextstateneg
  PosNegReindex(vitpos,vitneg),nextstate
end

function Arrays.return_value(
  f::Broadcasting{<:PosNegParamReindex},
  x::Union{Number,AbstractArray{<:Number}}...)

  v = return_value(Broadcasting(testitem(f.f)),x...)
  array = Vector{typeof(v)}(undef,length(f.f))
  for (i,fi) = enumerate(f.f)
    array[i] = return_value(Broadcasting(fi),x...)
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
  for (i,fi) = enumerate(f.f)
    cache[i] = return_cache(Broadcasting(fi),x...)
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{<:PosNegParamReindex},
  x::Union{Number,AbstractArray{<:Number}}...)

  cx,array = cache
  @inbounds for (i,fi) = enumerate(f.f)
    array[i] = evaluate!(cx[i],Broadcasting(fi),x...)
  end
  array
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{<:PosNegParamReindex},
  x::AbstractArray{<:Number})

  cx,array = cache
  @inbounds for (i,fi) = enumerate(f.f)
    array[i] = evaluate!(cx[i],Broadcasting(fi),x)
  end
  array
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{<:PosNegParamReindex},
  x::Number...)

  cx,array = cache
  @inbounds for (i,fi) = enumerate(f.f)
    array[i] = evaluate!(cx[i],Broadcasting(fi),x...)
  end
  array
end

for T in (:ParamReindex,:PosNegParamReindex)
  @eval begin
    function Arrays.return_value(k::$T,j::Integer)
      v = return_value(testitem(k),j)
      array = Vector{typeof(v)}(undef,length(k))
      for (i,ki) = enumerate(k)
        array[i] = return_value(ki,j)
      end
      ParamArray(array)
    end

    function Arrays.return_cache(k::$T,j::Integer)
      c = return_cache(testitem(k),j)
      a = evaluate!(c,testitem(k),j)
      cache = Vector{typeof(c)}(undef,length(k))
      array = Vector{typeof(a)}(undef,length(k))
      for (i,ki) = enumerate(k)
        cache[i] = return_cache(ki,j)
      end
      cache,ParamArray(array)
    end

    function Arrays.evaluate!(cache,k::$T,j::Integer)
      cx,array = cache
      @inbounds for (i,ki) = enumerate(k)
        array[i] = evaluate!(cx[i],ki,j)
      end
      array
    end

    function Arrays.evaluate(k::$T,j::Integer)
      array = map(k) do ki
        evaluate(ki,j)
      end
      ParamArray(array)
    end
  end
end
