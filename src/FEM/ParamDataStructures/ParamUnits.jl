abstract type ParamUnit{A} end

Base.size(b::ParamUnit) = tfill(param_length(b),Val{ndims(b)}())
Base.length(b::ParamUnit) = param_length(b)^ndims(b)
Base.eltype(::Type{<:ParamUnit{A}}) where A = A
Base.eltype(b::ParamUnit{A}) where A = A
Base.ndims(b::ParamUnit{A}) where A = ndims(A)
Base.ndims(::Type{<:ParamUnit{A}}) where A = ndims(A)

Arrays.testitem(b::ParamUnit) = param_getindex(b,1)

innersize(b::ParamUnit) = size(testitem(b))
innerlength(b::ParamUnit) = prod(innersize(b))

function Base.:≈(a::AbstractArray{<:ParamUnit},b::AbstractArray{<:ParamUnit})
  all(z->z[1]≈z[2],zip(a,b))
end

struct GenericParamUnit{A} <: ParamUnit{A}
  data::Vector{A}
end

function Base.getindex(b::GenericParamUnit{A},i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.data[iblock]
  else
    testvalue(A)
  end
end

function Base.setindex!(b::GenericParamUnit{A},v,i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.data[iblock] = v
  end
end

get_param_data(b::GenericParamUnit) = b.data
param_length(b::GenericParamUnit) = length(b.data)
param_getindex(b::GenericParamUnit,i::Integer) = b.data[i]
param_setindex!(b::GenericParamUnit,v,i::Integer) = (b.data[i]=v)

function get_param_entry!(v::AbstractVector,b::GenericParamUnit,i...)
  for k in eachindex(v)
    @inbounds v[k] = b.data[k][i...]
  end
  v
end

Base.copy(a::GenericParamUnit) = GenericParamUnit(copy(a.data))

function Base.copyto!(a::GenericParamUnit,b::GenericParamUnit)
  @check size(a) == size(b)
  for i in eachindex(a.data)
    fill!(a.data[i],zero(eltype(a.data[i])))
    copyto!(a.data[i],b.data[i])
  end
  a
end

function Base.:≈(a::GenericParamUnit,b::GenericParamUnit)
  if size(a) != size(b)
    return false
  end
  for i in eachindex(a.data)
    if !(a.data[i] ≈ b.data[i])
      return false
    end
  end
  true
end

function Base.:(==)(a::GenericParamUnit,b::GenericParamUnit)
  if size(a) != size(b)
    return false
  end
  for i in eachindex(a.data)
    if a.data[i] != b.data[i]
      return false
    end
  end
  true
end

function Arrays.testvalue(::Type{GenericParamUnit{A}}) where A
  GenericParamUnit([testvalue(A)])
end

function Arrays.CachedArray(a::GenericParamUnit)
  ai = testitem(a)
  ci = CachedArray(ai)
  data = Vector{typeof(ci)}(undef,param_length(a))
  for i in eachindex(a.data)
    data[i] = CachedArray(a.data[i])
  end
  GenericParamUnit(data)
end

function Fields.unwrap_cached_array(a::GenericParamUnit)
  cache = return_cache(Fields.unwrap_cached_array,a)
  evaluate!(cache,Fields.unwrap_cached_array,a)
end

function Arrays.return_cache(::typeof(Fields.unwrap_cached_array),a::GenericParamUnit)
  ai = testitem(a)
  ci = return_cache(Fields.unwrap_cached_array,ai)
  ri = evaluate!(ci,Fields.unwrap_cached_array,ai)
  c = Vector{typeof(ci)}(undef,length(a.data))
  data = Vector{typeof(ri)}(undef,length(a.data))
  for i in eachindex(a.data)
    c[i] = return_cache(Fields.unwrap_cached_array,a.data[i])
  end
  GenericParamUnit(data),c
end

function Arrays.evaluate!(cache,::typeof(Fields.unwrap_cached_array),a::GenericParamUnit)
  r,c = cache
  for i in eachindex(a.data)
    r.data[i] = evaluate!(c[i],Fields.unwrap_cached_array,a.data[i])
  end
  r
end

###################### trivial case ######################

struct TrivialParamUnit{A} <: ParamUnit{A}
  data::A
  plength::Int
end

function TrivialParamUnit(data::Any)
  plength = 1
  TrivialParamUnit(data,plength)
end

function Base.getindex(b::TrivialParamUnit{A},i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.data
  else
    testvalue(A)
  end
end

function Base.setindex!(b::TrivialParamUnit{A},v,i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.data = v
  end
end

get_param_data(b::TrivialParamUnit) = Fill(b.data,b.plength)
param_length(b::TrivialParamUnit) = b.plength
param_getindex(b::TrivialParamUnit,i::Integer) = b.data
param_setindex!(b::TrivialParamUnit,v,i::Integer) = copyto!(b.data,v)

function get_param_entry!(v::AbstractVector,b::TrivialParamUnit,i...)
  vk = b.data[k][i...]
  fill!(v,vk)
end

Base.copy(a::TrivialParamUnit) = TrivialParamUnit(copy(a.data),a.plength)

Base.copyto!(a::TrivialParamUnit,b::TrivialParamUnit) = copyto!(a.data,b.data)

function Base.:≈(a::TrivialParamUnit,b::TrivialParamUnit)
  if size(a) != size(b)
    return false
  end
  a.data ≈ b.data
end

function Base.:(==)(a::TrivialParamUnit,b::TrivialParamUnit)
  if size(a) != size(b)
    return false
  end
  a.data == b.data
end

function Arrays.testvalue(::Type{TrivialParamUnit{A}}) where A
  TrivialParamUnit(testvalue(A),1)
end

function Arrays.CachedArray(a::TrivialParamUnit)
  TrivialParamUnit(CachedArray(a.data),a.plength)
end

function Fields.unwrap_cached_array(a::TrivialParamUnit)
  TrivialParamUnit(Fields.unwrap_cached_array(a.data),a.plength)
end

###################### trivial case ######################

function Arrays.return_cache(f::GenericParamUnit,x)
  fi = testitem(f)
  li = return_cache(fi,x)
  fix = evaluate!(li,fi,x)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(f.data[i],x)
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,f::GenericParamUnit,x)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],f.data[i],x)
  end
  g
end

function Fields.linear_combination(u::GenericParamUnit,f::GenericParamUnit)
  @check size(u) == size(f)
  fi = testitem(f)
  ui = testitem(u)
  ufi = linear_combination(ui,fi)
  g = Vector{typeof(ufi)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = linear_combination(u.data[i],f.data[i])
  end
  GenericParamUnit(g)
end

function Fields.linear_combination(u::GenericParamUnit,f::AbstractVector{<:Field})
  ufi = linear_combination(testitem(u),f)
  g = Vector{typeof(ufi)}(undef,param_length(u))
  @inbounds for i in param_eachindex(u)
    g[i] = linear_combination(param_getindex(u,i),f)
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::LinearCombinationMap,u::GenericParamUnit,fx::AbstractArray)
  ui = testitem(u)
  li = return_cache(k,ui,fx)
  ufxi = evaluate!(li,k,ui,fx)
  l = Vector{typeof(li)}(undef,length(u.data))
  g = Vector{typeof(ufxi)}(undef,length(u.data))
  for i in eachindex(u.data)
    l[i] = return_cache(k,u.data[i],fx)
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::GenericParamUnit,fx::AbstractArray)
  g,l = cache
  for i in eachindex(u.data)
    g.data[i] = evaluate!(l[i],k,u.data[i],fx)
  end
  g
end

function Arrays.return_cache(k::LinearCombinationMap,u::GenericParamUnit,fx::GenericParamUnit)
  fxi = testitem(fx)
  ui = testitem(u)
  li = return_cache(k,ui,fxi)
  ufxi = evaluate!(li,k,ui,fxi)
  l = Vector{typeof(li)}(undef,length(fx.data))
  g = Vector{typeof(ufxi)}(undef,length(fx.data))
  for i in eachindex(fx.data)
    l[i] = return_cache(k,u.data[i],fx.data[i])
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::GenericParamUnit,fx::GenericParamUnit)
  g,l = cache
  for i in eachindex(fx.data)
    g.data[i] = evaluate!(l[i],k,u.data[i],fx.data[i])
  end
  g
end

function Base.transpose(f::GenericParamUnit)
  fi = testitem(f)
  fit = transpose(fi)
  g = Vector{typeof(fit)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = transpose(f.data[i])
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::Fields.TransposeMap,f::GenericParamUnit)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i])
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::Fields.TransposeMap,f::GenericParamUnit)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i])
  end
  g
end

function Fields.integrate(f::GenericParamUnit,args...)
  fi = testitem(f)
  intfi = integrate(fi,args...)
  g = Vector{typeof(intfi)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = integrate(f.data[i],args...)
  end
  GenericParamUnit(g)
end

function Arrays.return_value(k::IntegrationMap,fx::GenericParamUnit,args...)
  fxi = testitem(fx)
  ufxi = return_value(k,fxi,args...)
  g = Vector{typeof(ufxi)}(undef,length(fx.data))
  for i in eachindex(fx.data)
    g[i] = return_value(k,fx.data[i],args...)
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::IntegrationMap,fx::GenericParamUnit,args...)
  fxi = testitem(fx)
  li = return_cache(k,fxi,args...)
  ufxi = evaluate!(li,k,fxi,args...)
  l = Vector{typeof(li)}(undef,length(fx.data))
  g = Vector{typeof(ufxi)}(undef,length(fx.data))
  for i in eachindex(fx.data)
    l[i] = return_cache(k,fx.data[i],args...)
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx::GenericParamUnit,args...)
  g,l = cache
  for i in eachindex(fx.data)
    g.data[i] = evaluate!(l[i],k,fx.data[i],args...)
  end
  g
end

function Arrays.return_value(k::Broadcasting,f::GenericParamUnit)
  fi = testitem(f)
  fix = return_value(k,fi)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(k,f.data[i])
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::Broadcasting,f::GenericParamUnit)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i])
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting,f::GenericParamUnit)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i])
  end
  g
end

function Arrays.return_value(k::Broadcasting{typeof(∘)},f::GenericParamUnit,h::Field)
  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(k,f.data[i],h)
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::GenericParamUnit,h::Field)
  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i],h)
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::GenericParamUnit,h::Field)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i],h)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:Operation},f::GenericParamUnit,h::Field)
  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(k,f.data[i],h)
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},f::GenericParamUnit,h::Field)
  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i],h)
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},f::GenericParamUnit,h::Field)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i],h)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:Operation},h::Field,f::GenericParamUnit)
  fi = testitem(f)
  fix = return_value(k,h,fi)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(k,h,f.data[i])
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},h::Field,f::GenericParamUnit)
  fi = testitem(f)
  li = return_cache(k,h,fi)
  fix = evaluate!(li,k,h,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,h,f.data[i])
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},h::Field,f::GenericParamUnit)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,h,f.data[i])
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:Operation},h::GenericParamUnit,f::GenericParamUnit)
  @check param_length(h) == param_length(f)
  hi = testitem(h)
  fi = testitem(f)
  fix = return_value(k,hi,fi)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(k,h.data[i],f.data[i])
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},h::GenericParamUnit,f::GenericParamUnit)
  @check param_length(h) == param_length(f)
  hi = testitem(h)
  fi = testitem(f)
  li = return_cache(k,hi,fi)
  fix = evaluate!(li,k,hi,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,h.data[i],f.data[i])
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},h::GenericParamUnit,f::GenericParamUnit)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,h.data[i],f.data[i])
  end
  g
end

const ParamOperation = Operation{<:AbstractParamFunction}

param_length(f::Broadcasting{<:ParamOperation}) = param_length(f.f)
param_getindex(f::Broadcasting{<:ParamOperation},i::Int) = Broadcasting(param_getindex(f.f,i))
Arrays.testitem(f::Broadcasting{<:ParamOperation}) = param_getindex(f,1)

function Arrays.return_value(k::Broadcasting{<:ParamOperation},f::GenericParamUnit,h::Field)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  fix = return_value(ki,fi,h)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(param_getindex(k,i),f.data[i],h)
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},f::GenericParamUnit,h::Field)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  li = return_cache(ki,fi,h)
  fix = evaluate!(li,ki,fi,h)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(param_getindex(k,i),f.data[i],h)
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ParamOperation},f::GenericParamUnit,h::Field)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],param_getindex(k,i),f.data[i],h)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ParamOperation},h::Field,f::GenericParamUnit)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  fix = return_value(ki,h,fi)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(param_getindex(k,i),h,f.data[i])
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},h::Field,f::GenericParamUnit)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  li = return_cache(ki,h,fi)
  fix = evaluate!(li,ki,h,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(param_getindex(k,i),h,f.data[i])
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ParamOperation},h::Field,f::GenericParamUnit)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],param_getindex(k,i),h,f.data[i])
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ParamOperation},h::GenericParamUnit,f::GenericParamUnit)
  @check param_length(k) == param_length(h) == param_length(f)
  ki = testitem(k)
  hi = testitem(h)
  fi = testitem(f)
  fix = return_value(ki,hi,fi)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(param_getindex(k,i),h.data[i],f.data[i])
  end
  GenericParamUnit(g)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},h::GenericParamUnit,f::GenericParamUnit)
  @check param_length(k) == param_length(h) == param_length(f)
  ki = testitem(k)
  hi = testitem(h)
  fi = testitem(f)
  li = return_cache(ki,hi,fi)
  fix = evaluate!(li,ki,hi,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(param_getindex(k,i),h.data[i],f.data[i])
  end
  GenericParamUnit(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ParamOperation},h::GenericParamUnit,f::GenericParamUnit)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],param_getindex(k,i),h.data[i],f.data[i])
  end
  g
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::GenericParamUnit,g::AbstractArray)
  fi = testitem(f)
  fix = return_value(k,fi,g)
  h = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    h[i] = return_value(k,f.data[i],g)
  end
  GenericParamUnit(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::GenericParamUnit,g::AbstractArray)
  fi = testitem(f)
  li = return_cache(k,fi,g)
  fix = evaluate!(li,k,fi,g)
  l = Vector{typeof(li)}(undef,length(f.data))
  h = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i],g)
  end
  GenericParamUnit(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::GenericParamUnit,g::AbstractArray)
  h,l = cache
  for i in eachindex(f.data)
    h.data[i] = evaluate!(l[i],k,f.data[i],g)
  end
  h
end

function Arrays.return_value(k::BroadcastingFieldOpMap,g::AbstractArray,f::GenericParamUnit)
  fi = testitem(f)
  fix = return_value(k,g,fi)
  h = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    h[i] = return_value(k,g,f.data[i])
  end
  GenericParamUnit(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,g::AbstractArray,f::GenericParamUnit)
  fi = testitem(f)
  li = return_cache(k,g,fi)
  fix = evaluate!(li,k,g,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  h = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,g,f.data[i])
  end
  GenericParamUnit(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,g::AbstractArray,f::GenericParamUnit)
  h,l = cache
  for i in eachindex(f.data)
    h.data[i] = evaluate!(l[i],k,g,f.data[i])
  end
  h
end

for op in (:+,:-,:*)
  @eval begin

    function Arrays.return_value(k::Broadcasting{typeof($op)},f::GenericParamUnit,g::GenericParamUnit)
      return_value(BroadcastingFieldOpMap($op),f,g)
    end

    function Arrays.return_cache(k::Broadcasting{typeof($op)},f::GenericParamUnit,g::GenericParamUnit)
      return_cache(BroadcastingFieldOpMap($op),f,g)
    end

    function Arrays.evaluate!(cache,k::Broadcasting{typeof($op)},f::GenericParamUnit,g::GenericParamUnit)
      evaluate!(cache,BroadcastingFieldOpMap($op),f,g)
    end

  end
end

function Arrays.return_value(k::Broadcasting{typeof(*)},f::Number,g::GenericParamUnit)
  gi = testitem(g)
  hi = return_value(k,f,gi)
  data = Vector{typeof(hi)}(undef,length(g.data))
  for i in eachindex(g.data)
    data[i] = return_value(k,f,g.data[i])
  end
  GenericParamUnit(data)
end

function Arrays.return_cache(k::Broadcasting{typeof(*)},f::Number,g::GenericParamUnit)
  gi = testitem(g)
  ci = return_cache(k,f,gi)
  hi = evaluate!(ci,k,f,gi)
  data = Vector{typeof(hi)}(undef,length(g.data))
  c = Vector{typeof(ci)}(undef,length(g.data))
  for i in eachindex(g.data)
    c[i] = return_cache(k,f,g.data[i])
  end
  GenericParamUnit(data),c
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(*)},f::Number,g::GenericParamUnit)
  r,c = cache
  for i in eachindex(g.data)
    r.data[i] = evaluate!(c[i],k,f,g.data[i])
  end
  r
end

function Arrays.return_value(k::Broadcasting{typeof(*)},f::GenericParamUnit,g::Number)
  evaluate(k,f,g)
end

function Arrays.return_cache(k::Broadcasting{typeof(*)},f::GenericParamUnit,g::Number)
  return_cache(k,g,f)
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(*)},f::GenericParamUnit,g::Number)
  evaluate!(cache,k,g,f)
end

function Arrays.return_value(k::BroadcastingFieldOpMap,a::ParamUnit...)
  evaluate(k,a...)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,a::TrivialParamUnit...)
  a1 = first(a)
  @notimplementedif any(ai->param_length(ai)!=param_length(a1),a)
  ais = map(ai->ai.data,a)
  ai = return_cache(k,ais...)
  bi = evaluate!(ci,k,ais...)
  TrivialParamUnit(bi,a1.plength),ai
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::TrivialParamUnit...)
  a1 = first(a)
  @notimplementedif any(ai->param_length(ai)!=param_length(a1),a)
  r,c = cache
  ais = map(ai->ai.data,a)
  copyto!(r.data,evaluate!(c,k,ais...))
  r
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::ParamUnit,g::ParamUnit)
  @notimplementedif param_length(f) != param_length(g)
  fi = testitem(f)
  gi = testitem(g)
  hi = return_value(k,fi,gi)
  a = Vector{typeof(hi)}(undef,param_length(f))
  fill!(a,hi)
  GenericParamUnit(a)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::ParamUnit,g::ParamUnit)
  @notimplementedif param_length(f) != param_length(g)
  fi = testitem(f)
  gi = testitem(g)
  ci = return_cache(k,fi,gi)
  hi = evaluate!(ci,k,fi,gi)
  a = Vector{typeof(hi)}(undef,param_length(f))
  b = Vector{typeof(ci)}(undef,param_length(f))
  for i in param_eachindex(f)
    b[i] = return_cache(k,param_getindex(f,i),param_getindex(g,i))
  end
  GenericParamUnit(a),b
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::ParamUnit,g::ParamUnit)
  @notimplementedif param_length(f) != param_length(g)
  a,b = cache
  for i in eachindex(f.data)
    a.data[i] = evaluate!(b[i],k,param_getindex(f,i),param_getindex(g,i))
  end
  a
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,a::ParamUnit...)
  a1 = first(a)
  @notimplementedif any(ai->param_length(ai)!=param_length(a1),a)
  ais = map(testitem,a)
  ci = return_cache(k,ais...)
  bi = evaluate!(ci,k,ais...)
  c = Vector{typeof(ci)}(undef,length(a1.data))
  data = Vector{typeof(bi)}(undef,length(a1.data))
  for i in eachindex(a1.data)
    _ais = map(ai->ai.data[i],a)
    c[i] = return_cache(k,_ais...)
  end
  GenericParamUnit(data),c
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::ParamUnit...)
  a1 = first(a)
  @notimplementedif any(ai->param_length(ai)!=param_length(a1),a)
  r,c = cache
  for i in eachindex(a1.data)
    ais = map(ai->ai.data[i],a)
    r.data[i] = evaluate!(c[i],k,ais...)
  end
  r
end

function Arrays.return_value(
  k::BroadcastingFieldOpMap,a::Union{ParamUnit,AbstractArray}...)
  evaluate(k,a...)
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,a::Union{ParamUnit,AbstractArray}...)

  return_cache(k,lazy_parameterize(a...)...)
end

function Arrays.evaluate!(
  cache,k::BroadcastingFieldOpMap,a::Union{ParamUnit,AbstractArray}...)

  evaluate!(cache,k,lazy_parameterize(a...)...)
end

for op in (:+,:-)
  @eval begin
    function $op(a::ParamUnit,b::ParamUnit)
      BroadcastingFieldOpMap($op)(a,b)
    end

    function $op(a::TrivialParamUnit,b::TrivialParamUnit)
      @check size(a) == size(b)
      c = $op(a.data,b.data)
      TrivialParamUnit(c,a.plength)
    end
  end
end

function Base.:*(a::Number,b::GenericParamUnit)
  bi = testitem(b)
  ci = a*bi
  data = Vector{typeof(ci)}(undef,length(b.data))
  for i in eachindex(b.data)
    data[i] = a*b.data[i]
  end
  GenericParamUnit(data)
end

function Base.:*(a::Number,b::TrivialParamUnit)
  TrivialParamUnit(a*b.data,b.plength)
end

function Base.:*(a::ParamUnit,b::Number)
  b*a
end

function Base.:*(a::TrivialParamUnit,b::TrivialParamUnit)
  @check param_length(a) == param_length(b)
  c = a.data*b.data
  TrivialParamUnit(c,a.plength)
end

function LinearAlgebra.mul!(c::TrivialParamUnit,a::TrivialParamUnit,b::TrivialParamUnit)
  mul!(c.data,a.data,b.data,1,0)
end

function LinearAlgebra.rmul!(a::TrivialParamUnit,β)
  rmul!(a.data,β)
end

function Arrays.return_value(::typeof(*),a::TrivialParamUnit,b::TrivialParamUnit)
  evaluate(*,a,b)
end

function Arrays.return_cache(::typeof(*),a::TrivialParamUnit,b::TrivialParamUnit)
  CachedArray(a*b)
end

function Arrays.evaluate!(cache,::typeof(*),a::TrivialParamUnit,b::TrivialParamUnit)
  Fields._setsize_mul!(cache,a.data,b.data)
  r = cache.data
  mul!(r,a.data,b.data)
  r
end

function Base.:*(a::ParamUnit,b::ParamUnit)
  @check param_length(a) == param_length(b)
  ai = testitem(a)
  bi = testitem(b)
  ri = ai*bi
  data = Vector{typeof(ri)}(undef,param_length(a))
  data[1] = ri
  for i in 2:param_length(a)
    data[i] = param_getindex(a,i)*param_getindex(b,i)
  end
  GenericParamUnit(data)
end

function Arrays.return_value(::typeof(*),a::ParamUnit,b::ParamUnit)
  @check param_length(a) == param_length(b)
  ai = testitem(a)
  bi = testitem(b)
  ri = return_value(*,ai,bi)
  data = Vector{typeof(ri)}(undef,param_length(a))
  GenericParamUnit(data)
end

function LinearAlgebra.rmul!(a::GenericParamUnit,β)
  for i in eachindex(a.data)
    rmul!(a.data[i],β)
  end
end

function Fields._zero_entries!(a::GenericParamUnit)
  for i in eachindex(a.data)
    Fields._zero_entries!(a.data[i])
  end
end

function LinearAlgebra.mul!(c::ParamUnit,a::ParamUnit,b::ParamUnit)
  Fields._zero_entries!(c)
  mul!(c,a,b,1,0)
end

function LinearAlgebra.mul!(
  c::ParamUnit,
  a::ParamUnit,
  b::ParamUnit,
  α::Number,β::Number)

  for i in eachindex(c.data)
    mul!(param_getindex(c,i),param_getindex(a,i),param_getindex(b,i),α,β)
  end
end

function Fields._setsize_mul!(c,a::ParamUnit,b::ParamUnit)
  for i in eachindex(c.data)
    Fields._setsize_mul!(param_getindex(c,i),param_getindex(a,i),param_getindex(b,i))
  end
end

function Arrays.return_cache(::typeof(*),a::ParamUnit,b::ParamUnit)
  c1 = CachedArray(a*b)
  c2 = return_cache(Fields.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::ParamUnit,b::ParamUnit)
  c1,c2 = cache
  Fields._setsize_mul!(c1,a,b)
  c = evaluate!(c2,Fields.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Fields._setsize_as!(d,a::GenericParamUnit)
  for i in eachindex(a.data)
    Fields._setsize_mul!(param_getindex(d,i),param_getindex(a,i))
  end
  d
end

function Arrays.return_value(k::MulAddMap,a::ParamUnit,b::ParamUnit,c::ParamUnit)
  x = return_value(*,a,b)
  return_value(+,x,c)
end

function Arrays.return_cache(k::MulAddMap,a::ParamUnit,b::ParamUnit,c::ParamUnit)
  c1 = CachedArray(a*b+c)
  c2 = return_cache(unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,k::MulAddMap,a::ParamUnit,b::ParamUnit,c::ParamUnit)
  c1,c2 = cache
  Fields._setsize_as!(c1,c)
  Fields._setsize_mul!(c1,a,b)
  d = evaluate!(c2,Fields.unwrap_cached_array,c1)
  copyto!(d,c)
  iszero(k.α) && isone(k.β) && return d
  mul!(d,a,b,k.α,k.β)
  d
end

# Autodiff related

function Arrays.return_cache(k::Arrays.ConfigMap{typeof(ForwardDiff.gradient)},x::GenericParamUnit)
  xi = testitem(x)
  fi = return_cache(k,xi)
  data = Vector{typeof(fi)}(undef,length(x.data))
  for i in eachindex(x.data)
    data[i] = return_cache(k,x.data[i])
  end
  GenericParamUnit(data)
end

function Arrays.return_cache(k::Arrays.ConfigMap{typeof(ForwardDiff.jacobian)},x::GenericParamUnit)
  xi = testitem(x)
  fi = return_cache(k,xi)
  data = Vector{typeof(fi)}(undef,length(x.data))
  for i in eachindex(x.data)
    data[i] = return_cache(k,x.data[i])
  end
  GenericParamUnit(data)
end

function Arrays.return_cache(k::Arrays.DualizeMap,x::GenericParamUnit)
  cfg = return_cache(Arrays.ConfigMap(k.f),x)
  xi = testitem(x)
  cfgi = testitem(cfg)
  xidual = evaluate!(cfgi,k,xi)
  data = Vector{typeof(xidual)}(undef,length(x.data))
  cfg,GenericParamUnit(data)
end

function Arrays.evaluate!(cache,k::Arrays.DualizeMap,x::GenericParamUnit)
  cfg,xdual = cache
  for i in eachindex(x.data)
    xdual.data[i] = evaluate!(cfg.data[i],k,x.data[i])
  end
  xdual
end

function Arrays.return_cache(k::Arrays.AutoDiffMap,ydual::GenericParamUnit,x,cfg::GenericParamUnit)
  yidual = testitem(ydual)
  xi = testitem(x)
  cfgi = testitem(cfg)
  ci = return_cache(k,yidual,xi,cfgi)
  ri = evaluate!(ci,k,yidual,xi,cfgi)
  c = Vector{typeof(ci)}(undef,length(ydual.data))
  data = Vector{typeof(ri)}(undef,length(ydual.data))
  for i in eachindex(ydual.data)
    c[i] = return_cache(k,ydual.data[i],x.data[i],cfg.data[i])
  end
  GenericParamUnit(data),c
end

function Arrays.evaluate!(cache,k::Arrays.AutoDiffMap,ydual::GenericParamUnit,x,cfg::GenericParamUnit)
  r,c = cache
  for i in eachindex(ydual.data)
    r.data[i] = evaluate!(c[i],k,ydual.data[i],x.data[i],cfg.data[i])
  end
  r
end

function Arrays.return_cache(k::CellData.ZeroVectorMap,a::TrivialParamUnit)
  c = return_cache(k,a.data)
  data = evaluate!(ci,k,a.data)
  TrivialParamUnit(data,v.plength),c
end

function Arrays.evaluate!(cache,k::CellData.ZeroVectorMap,a::TrivialParamUnit)
  r,c = cache
  copyto!(r.data,evaluate!(c[i],k,a.data))
  r
end

function Arrays.return_cache(k::CellData.ZeroVectorMap,a::GenericParamUnit)
  ai = testitem(a)
  ci = return_cache(k,ai)
  vi = evaluate!(ci,k,ai)
  c = Vector{typeof(ci)}(undef,param_length(a))
  data = Vector{typeof(vi)}(undef,param_length(a))
  for i in 1:param_length(a)
    c[i] = return_cache(k,param_getindex(a,i))
  end
  GenericParamUnit(data),c
end

function Arrays.evaluate!(cache,k::CellData.ZeroVectorMap,a::GenericParamUnit)
  r,c = cache
  for i in eachindex(ydual.data)
    r.data[i] = evaluate!(c[i],k,a.data[i])
  end
  r
end

# cell datas

function Geometry._cache_compress(data::ParamUnit)
  c1 = CachedArray(deepcopy(data))
  c2 = return_cache(Fields.unwrap_cached_array,c1)
  c1,c2
end

function Geometry._setempty_compress!(a::TrivialParamUnit)
  Geometry._setempty_compress!(a.data)
end

function Geometry._setempty_compress!(a::GenericParamUnit)
  for i in eachindex(a.data)
    Geometry._setempty_compress!(a.data[i])
  end
end

function Geometry._uncached_compress!(c1::ParamUnit,c2)
  evaluate!(c2,Fields.unwrap_cached_array,c1)
end

function Geometry._setsize_compress!(a::TrivialParamUnit,b::TrivialParamUnit)
  Geometry._setsize_compress!(a.data,b.data)
end

function Geometry._setsize_compress!(a::ParamUnit,b::ParamUnit)
  @check size(a) == size(b)
  for i in param_eachindex(a)
    Geometry._setsize_compress!(param_getindex(a,i),param_getindex(b,i))
  end
end

function Geometry._copyto_compress!(a::TrivialParamUnit,b::TrivialParamUnit)
  Geometry._copyto_compress!(a.data,b.data)
end

function Geometry._copyto_compress!(a::ParamUnit,b::ParamUnit)
  @check size(a) == size(b)
  for i in param_eachindex(a)
    Geometry._copyto_compress!(param_getindex(a,i),param_getindex(b,i))
  end
end

function Geometry._addto_compress!(a::TrivialParamUnit,b::TrivialParamUnit)
  Geometry._addto_compress!(a.data,b.data)
end

function Geometry._addto_compress!(a::ParamUnit,b::ParamUnit)
  @check size(a) == size(b)
  for i in param_eachindex(a)
    Geometry._addto_compress!(param_getindex(a,i),param_getindex(b,i))
  end
end

function Geometry._similar_empty(val::TrivialParamUnit)
  TrivialParamUnit(Geometry._similar_empty(val.data),val.plength)
end

function Geometry._similar_empty(val::GenericParamUnit)
  a = deepcopy(val)
  for i in eachindex(a)
    a.data[i] = Geometry._similar_empty(a.data[i])
  end
  a
end

function Geometry.pos_neg_data(
  ipos_to_val::AbstractArray{<:ParamUnit},i_to_iposneg::PosNegPartition)
  nineg = length(i_to_iposneg.ineg_to_i)
  val = testitem(ipos_to_val)
  void = Geometry._similar_empty(val)
  ineg_to_val = Fill(void,nineg)
  ipos_to_val,ineg_to_val
end

# reference FEs

function Arrays.return_cache(b::LagrangianDofBasis,f::GenericParamUnit)
  fi = testitem(f)
  ci = return_cache(b,fi)
  ri = evaluate!(ci,b,fi)
  c = Vector{typeof(ci)}(undef,length(f.data))
  data = Vector{typeof(ri)}(undef,length(f.data))
  for i in eachindex(f.data)
    c[i] = return_cache(b,f.data[i])
  end
  GenericParamUnit(data),c
end

function Arrays.evaluate!(cache,b::LagrangianDofBasis,f::GenericParamUnit)
  r,c = cache
  for i in eachindex(f.data)
    r.data[i] = evaluate!(c[i],b,f.data[i])
  end
  r
end

# block map interface

function Arrays.return_value(k::BroadcastingFieldOpMap,f::ArrayBlock{A,N},h::ParamUnit) where {A,N}
  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = Array{typeof(fix),N}(undef,size(f.array))
  for i in eachindex(f.array)
    if f.touched[i]
      g[i] = return_value(k,f.array[i],h)
    end
  end
  ArrayBlock(g,f.touched)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::ArrayBlock{A,N},h::ParamUnit) where {A,N}
  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = Array{typeof(li),N}(undef,size(f.array))
  g = Array{typeof(fix),N}(undef,size(f.array))
  for i in eachindex(f.array)
    if f.touched[i]
      l[i] = return_cache(k,f.array[i],h)
    end
  end
  ArrayBlock(g,f.touched),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::ArrayBlock,h::ParamUnit)
  g,l = cache
  @check g.touched == f.touched
  for i in eachindex(f.array)
    if f.touched[i]
      g.array[i] = evaluate!(l[i],k,f.array[i],h)
    end
  end
  g
end

function Arrays.return_value(k::BroadcastingFieldOpMap,h::ParamUnit,f::ArrayBlock{A,N}) where {A,N}
  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = Array{typeof(fix),N}(undef,size(f.array))
  for i in eachindex(f.array)
    if f.touched[i]
      g[i] = return_value(k,h,f.array[i])
    end
  end
  ArrayBlock(g,f.touched)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,h::ParamUnit,f::ArrayBlock{A,N}) where {A,N}
  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = Array{typeof(li),N}(undef,size(f.array))
  g = Array{typeof(fix),N}(undef,size(f.array))
  for i in eachindex(f.array)
    if f.touched[i]
      l[i] = return_cache(k,h,f.array[i])
    end
  end
  ArrayBlock(g,f.touched),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,h::ParamUnit,f::ArrayBlock)
  g,l = cache
  @check g.touched == f.touched
  for i in eachindex(f.array)
    if f.touched[i]
      g.array[i] = evaluate!(l[i],k,h,f.array[i])
    end
  end
  g
end

# for S in (:ParamUnit,:AbstractArray)
#   for T in (:ParamUnit,:AbstractArray)
#     (S == :AbstractArray && T == :AbstractArray) && continue
#     @eval begin
#       function Arrays.return_cache(
#         k::BroadcastingFieldOpMap,
#         f::ArrayBlock{<:$S,N},
#         g::ArrayBlock{<:$T,N}
#         ) where N

#         @notimplementedif size(f) != size(g)
#         fi = testvalue(testitem(f))
#         gi = testvalue(testitem(g))
#         ci = return_cache(k,fi,gi)
#         hi = evaluate!(ci,k,fi,gi)
#         m = Fields.ZeroBlockMap()
#         a = Array{typeof(hi),N}(undef,size(f.array))
#         b = Array{typeof(ci),N}(undef,size(f.array))
#         zf = Array{typeof(return_cache(m,fi,gi))}(undef,size(f.array))
#         zg = Array{typeof(return_cache(m,gi,fi))}(undef,size(f.array))
#         t = map(|,f.touched,g.touched)
#         for i in eachindex(f.array)
#           if f.touched[i] && g.touched[i]
#             b[i] = return_cache(k,f.array[i],g.array[i])
#           elseif f.touched[i]
#             _fi = f.array[i]
#             zg[i] = return_cache(m,gi,_fi)
#             _gi = evaluate!(zg[i],m,gi,_fi)
#             b[i] = return_cache(k,_fi,_gi)
#           elseif g.touched[i]
#             _gi = g.array[i]
#             zf[i] = return_cache(m,fi,_gi)
#             _fi = evaluate!(zf[i],m,fi,_gi)
#             b[i] = return_cache(k,_fi,_gi)
#           end
#         end
#         ArrayBlock(a,t),b,zf,zg
#       end

#       function Arrays.return_cache(
#         k::BroadcastingFieldOpMap,
#         f::ArrayBlock{<:$S,1},
#         g::ArrayBlock{<:$T,2}
#         )

#         fi = testvalue(testitem(f))
#         gi = testvalue(testitem(g))
#         ci = return_cache(k,fi,gi)
#         hi = evaluate!(ci,k,fi,gi)
#         @check size(g.array,1) == 1 || size(g.array,2) == 0
#         s = (size(f.array,1),size(g.array,2))
#         a = Array{typeof(hi),2}(undef,s)
#         b = Array{typeof(ci),2}(undef,s)
#         t = fill(false,s)
#         for j in 1:s[2]
#           for i in 1:s[1]
#             if f.touched[i] && g.touched[1,j]
#               t[i,j] = true
#               b[i,j] = return_cache(k,f.array[i],g.array[1,j])
#             end
#           end
#         end
#         ArrayBlock(a,t),b
#       end

#       function Arrays.return_cache(
#         k::BroadcastingFieldOpMap,
#         f::ArrayBlock{<:$S,2},
#         g::ArrayBlock{<:$T,1}
#         )

#         fi = testvalue(testitem(f))
#         gi = testvalue(testitem(g))
#         ci = return_cache(k,fi,gi)
#         hi = evaluate!(ci,k,fi,gi)
#         @check size(f.array,1) == 1 || size(f.array,2) == 0
#         s = (size(g.array,1),size(f.array,2))
#         a = Array{typeof(hi),2}(undef,s)
#         b = Array{typeof(ci),2}(undef,s)
#         t = fill(false,s)
#         for j in 1:s[2]
#           for i in 1:s[1]
#             if f.touched[1,j] && g.touched[i]
#               t[i,j] = true
#               b[i,j] = return_cache(k,f.array[1,j],g.array[i])
#             end
#           end
#         end
#         ArrayBlock(a,t),b
#       end
#     end
#   end
# end

for A in (:ArrayBlock,:AbstractArray)
  for B in (:ArrayBlock,:AbstractArray)
    for C in (:ArrayBlock,:AbstractArray)
      if !(A == B == C)
        @eval begin
          function Arrays.evaluate!(cache,k::Fields.BroadcastingFieldOpMap,a::$A,b::$B,c::$C)
            function _replace_nz_blocks!(cache::ArrayBlock,vali::AbstractArray)
              for i in eachindex(cache.array)
                if cache.touched[i]
                  cache.array[i] = vali
                end
              end
              cache
            end

            function _replace_nz_blocks!(cache::ArrayBlock,val::ArrayBlock)
              for i in eachindex(cache.array)
                if cache.touched[i]
                  cache.array[i] = val.array[i]
                end
              end
              cache
            end

            eval_cache,replace_cache = cache
            cachea,cacheb,cachec = replace_cache

            _replace_nz_blocks!(cachea,a)
            _replace_nz_blocks!(cacheb,b)
            _replace_nz_blocks!(cachec,c)

            evaluate!(eval_cache,k,cachea,cacheb,cachec)
          end
        end
      end
      for D in (:ArrayBlock,:AbstractArray)
        if !(A == B == C == D)
          @eval begin
            function Arrays.evaluate!(cache,k::Fields.BroadcastingFieldOpMap,a::$A,b::$B,c::$C,d::$D)
              function _replace_nz_blocks!(cache::ArrayBlock,vali::AbstractArray)
                for i in eachindex(cache.array)
                  if cache.touched[i]
                    cache.array[i] = vali
                  end
                end
                cache
              end

              function _replace_nz_blocks!(cache::ArrayBlock,val::ArrayBlock)
                for i in eachindex(cache.array)
                  if cache.touched[i]
                    cache.array[i] = val.array[i]
                  end
                end
                cache
              end

              eval_cache,replace_cache = cache
              cachea,cacheb,cachec,cached = replace_cache

              _replace_nz_blocks!(cachea,a)
              _replace_nz_blocks!(cacheb,b)
              _replace_nz_blocks!(cachec,c)
              _replace_nz_blocks!(cached,d)

              evaluate!(eval_cache,k,cachea,cacheb,cachec,cached)
            end
          end
        end
      end
    end
  end
end

for T in (:ParamUnit,:AbstractArray,:Nothing), S in (:ParamUnit,:AbstractArray)
  (T∈(:AbstractArray,:Nothing) && S==:AbstractArray) && continue
  @eval begin
    function Arrays.return_cache(k::Fields.ZeroBlockMap,_h::$T,_f::$S)
      h,f = _parameterize(_h,_f)
      hi = testitem(h)
      fi = testitem(f)
      ci = return_cache(k,hi,fi)
      ri = evaluate!(ci,k,hi,fi)
      c = Vector{typeof(ci)}(undef,param_length(f))
      data = Vector{typeof(ri)}(undef,param_length(f))
      for i in param_eachindex(f)
        c[i] = return_cache(k,param_getindex(h,i),param_getindex(f,i))
      end
      GenericParamUnit(data),c
    end
  end
end

function Arrays.evaluate!(cache::Tuple,k::Fields.ZeroBlockMap,f,h::AbstractArray)
  g,l = cache
  for i in eachindex(g.data)
    g.data[i] = evaluate!(l[i],k,f,h)
  end
  g
end

function Arrays.evaluate!(cache::Tuple,k::Fields.ZeroBlockMap,_f::ParamUnit,h::AbstractArray)
  g,l = cache
  f = _parameterize(_f,length(g.data))
  for i in eachindex(g.data)
    g.data[i] = evaluate!(l[i],k,param_getindex(f,i),h)
  end
  g
end

function Arrays.evaluate!(cache::Tuple,k::Fields.ZeroBlockMap,f,_h::ParamUnit)
  g,l = cache
  h = _parameterize(_h,length(g.data))
  for i in eachindex(g.data)
    g.data[i] = evaluate!(l[i],k,f,param_getindex(h,i))
  end
  g
end

function Arrays.evaluate!(cache::Tuple,k::Fields.ZeroBlockMap,_f::ParamUnit,_h::ParamUnit)
  g,l = cache
  f = _parameterize(_f,length(g.data))
  h = _parameterize(_h,length(g.data))
  for i in eachindex(g.data)
    g.data[i] = evaluate!(l[i],k,param_getindex(f,i),param_getindex(h,i))
  end
  g
end

# constructors

function lazy_parameterize(a::ParamUnit,plength::Integer=param_length(a))
  TrivialParamUnit(testitem(a),plength)
end

function lazy_parameterize(a::Union{AbstractArray{<:Number},Nothing,Field,AbstractArray{<:Field}},plength::Integer)
  TrivialParamUnit(a,plength)
end

function local_parameterize(a::Union{AbstractArray{<:Number},Nothing,Field,AbstractArray{<:Field}},plength::Integer)
  data = Vector{typeof(a)}(undef,plength)
  @inbounds for i in 1:plength
    data[i] = copy(a)
  end
  GenericParamUnit(data)
end

function local_parameterize(a::AbstractArray{<:AbstractArray},plength::Integer)
  @check length(a) == plength
  GenericParamUnit(a)
end

function local_parameterize(a::ParamUnit,plength::Integer)
  @check param_length(a) == plength
  a
end

function Fields.GenericField(f::AbstractParamFunction)
  GenericParamUnit(map(i -> GenericField(f[i]),1:length(f)))
end

# need to correct this function

for T in (:ParamUnit,:(ArrayBlock{<:ParamUnit}))
  @eval begin
    function CellData.add_contribution!(
      a::DomainContribution,
      trian::Triangulation,
      b::AbstractArray{<:$T},
      op=+)

      if haskey(a.dict,trian)
        a.dict[trian] = lazy_map(Broadcasting(op),a.dict[trian],b)
      else
        if op == +
         a.dict[trian] = b
        else
         a.dict[trian] = lazy_map(Broadcasting(op),b)
        end
      end
      a
    end
  end
end

# utils

_is_testvalue(f::ParamUnit) = param_length(f)==1 && innerlength(f)==0

function _parameterize(_h::ParamUnit,_f::ParamUnit)
  istest_h = _is_testvalue(_h)
  istest_f = _is_testvalue(_f)
  if istest_h==istest_f
    h,f = _h,_f
  elseif !istest_h && istest_f
    h,f = _h,lazy_parameterize(_f,param_length(_h))
  elseif istest_h && !istest_f
    h,f = lazy_parameterize(_h,param_length(_f)),_f
  end
  return h,f
end

function _parameterize(_h::ParamUnit,_f)
  _h,lazy_parameterize(_f,param_length(_h))
end

function _parameterize(_h,_f::ParamUnit)
  lazy_parameterize(_h,param_length(_f)),_f
end

function _parameterize(_f::ParamUnit,plength::Int)
  if param_length(_f) == plength
    return _f
  else
    @check param_length(_f) == 1
    return lazy_parameterize(_f,plength)
  end
end
