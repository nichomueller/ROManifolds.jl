function Arrays.return_cache(
  f::AbstractArray{T},
  x::AbstractArray{<:Point}) where T<:PTField

  S = return_type(testitem(f),testitem(x))
  r = SupportQuantity(zeros(S,(size(x)...,size(f)...)))
  cr = CachedArray(r)
  if isconcretetype(T)
    cf = return_cache(f,testitem(x))
  else
    cf = nothing
  end
  cr,cf
end

function Arrays.evaluate!(c,f::AbstractArray{T},x::AbstractArray{<:Point}) where T<:PTField
  cr,cf = c
  setsize!(cr,(size(x)...,size(f)...))
  r = SupportQuantity(cr.array)
  if isconcretetype(T)
    for i in eachindex(f)
      fxi = evaluate!(cf,f[i],x)
      for j in CartesianIndices(x)
        r[j,i] = fxi[j]
      end
    end
  else
    for i in eachindex(f)
      for j in eachindex(x)
        r[j,i] = evaluate(f[i],x[j])
      end
    end
  end
  r
end

function Arrays.return_cache(
  f::Fields.BroadcastingFieldOpMap,
  a::SupportQuantity,
  b::AbstractMatrix)

  return_cache(f,first(a.array),b)
end

function Arrays.return_cache(
  f::Fields.BroadcastingFieldOpMap,
  b::AbstractMatrix,
  a::SupportQuantity)

  return_cache(f,b,first(a.array))
end

function Arrays.evaluate!(
  cache,
  f::Fields.BroadcastingFieldOpMap,
  a::SupportQuantity,
  b::AbstractMatrix)

  @check size(a,1) == size(b,1)
  np, ni = size(b)
  setsize!(cache,(np,ni))
  r = cache.array
  for p in 1:np
    ap = a[p]
    for i in 1:ni
      r[p,i] = f.op(ap,b[p,i])
    end
  end
  r
end

function Arrays.evaluate!(
  cache,
  f::Fields.BroadcastingFieldOpMap,
  b::AbstractMatrix,
  a::SupportQuantity)

  @check size(a,1) == size(b,1)
  np, ni = size(b)
  setsize!(cache,(np,ni))
  r = cache.array
  for p in 1:np
    ap = a[p]
    for i in 1:ni
      r[p,i] = f.op(b[p,i],ap)
    end
  end
  r
end

function Arrays.evaluate!(
  cache,
  f::Fields.BroadcastingFieldOpMap,
  a::SupportQuantity,
  b::AbstractArray{S,3} where S)

  @check size(a,1) == size(b,1)
  np,ni = size(a)
  nj = size(b,2)
  setsize!(cache,(np,ni,nj))
  r = cache.array
  for j in 1:nj
    for p in 1:np
      bpj = b[p,j]
      for i in 1:ni
        r[p,i,j] = f.op(a[p,i],bpj)
      end
    end
  end
  r
end

function Arrays.evaluate!(
  cache,
  f::Fields.BroadcastingFieldOpMap,
  b::AbstractArray{S,3} where S,
  a::SupportQuantity)

  @check size(a,1) == size(b,1)
  np, ni = size(a)
  nj = size(b,2)
  setsize!(cache,(np,ni,nj))
  r = cache.array
  for p in 1:np
    for j in 1:nj
      bpj = b[p,j]
      for i in 1:ni
        r[p,i,j] = f.op(bpj,a[p,i])
      end
    end
  end
  r
end
