abstract type ParamBlock{A} end

Base.size(b::ParamBlock) = tfill(param_length(b),Val{ndims(b)}())
Base.length(b::ParamBlock) = param_length(b)^ndims(b)
Base.eltype(::Type{<:ParamBlock{A}}) where A = A
Base.eltype(b::ParamBlock{A}) where A = A
Base.ndims(b::ParamBlock{A}) where A = ndims(A)
Base.ndims(::Type{<:ParamBlock{A}}) where A = ndims(A)

Arrays.testitem(b::ParamBlock) = param_getindex(b,1)

innersize(b::ParamBlock) = size(testitem(b))
innerlength(b::ParamBlock) = prod(innersize(b))

function Base.:≈(a::AbstractArray{<:ParamBlock},b::AbstractArray{<:ParamBlock})
  all(z->z[1]≈z[2],zip(a,b))
end

struct GenericParamBlock{A} <: ParamBlock{A}
  data::Vector{A}
end

function Base.getindex(b::GenericParamBlock{A},i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.data[iblock]
  else
    testvalue(A)
  end
end

function Base.setindex!(b::GenericParamBlock{A},v,i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.data[iblock] = v
  end
end

get_param_data(b::GenericParamBlock) = b.data
param_length(b::GenericParamBlock) = length(b.data)
param_getindex(b::GenericParamBlock,i::Integer) = b.data[i]
param_setindex!(b::GenericParamBlock,v,i::Integer) = (b.data[i]=v)

function get_param_entry!(v::AbstractVector,b::GenericParamBlock,i...)
  for k in eachindex(v)
    @inbounds v[k] = b.data[k][i...]
  end
  v
end

Base.copy(a::GenericParamBlock) = GenericParamBlock(copy(a.data))

function Base.copyto!(a::GenericParamBlock,b::GenericParamBlock)
  @check size(a) == size(b)
  for i in eachindex(a.data)
    fill!(a.data[i],zero(eltype(a.data[i])))
    copyto!(a.data[i],b.data[i])
  end
  a
end

function Base.:≈(a::GenericParamBlock,b::GenericParamBlock)
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

function Base.:(==)(a::GenericParamBlock,b::GenericParamBlock)
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

function Arrays.testvalue(a::GenericParamBlock{A}) where A
  v = testvalue(A)
  data = Vector{typeof(v)}(undef,param_length(a))
  for i in 1:param_length(a)
    data[i] = copy(v)
  end
  GenericParamBlock(data)
end

# this one misses the param length
function Arrays.testvalue(::Type{GenericParamBlock{A}}) where A
  GenericParamBlock([testvalue(A)])
end

function Arrays.CachedArray(a::GenericParamBlock)
  ai = testitem(a)
  ci = CachedArray(ai)
  data = Vector{typeof(ci)}(undef,param_length(a))
  for i in eachindex(a.data)
    data[i] = CachedArray(a.data[i])
  end
  GenericParamBlock(data)
end

function Fields.unwrap_cached_array(a::GenericParamBlock)
  cache = return_cache(Fields.unwrap_cached_array,a)
  evaluate!(cache,Fields.unwrap_cached_array,a)
end

function Arrays.return_cache(::typeof(Fields.unwrap_cached_array),a::GenericParamBlock)
  ai = testitem(a)
  ci = return_cache(Fields.unwrap_cached_array,ai)
  ri = evaluate!(ci,Fields.unwrap_cached_array,ai)
  c = Vector{typeof(ci)}(undef,length(a.data))
  data = Vector{typeof(ri)}(undef,length(a.data))
  for i in eachindex(a.data)
    c[i] = return_cache(Fields.unwrap_cached_array,a.data[i])
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,::typeof(Fields.unwrap_cached_array),a::GenericParamBlock)
  r,c = cache
  for i in eachindex(a.data)
    r.data[i] = evaluate!(c[i],Fields.unwrap_cached_array,a.data[i])
  end
  r
end

###################### trivial case ######################

struct TrivialParamBlock{A} <: ParamBlock{A}
  data::A
  plength::Int
end

function TrivialParamBlock(data::Any)
  plength = 1
  TrivialParamBlock(data,plength)
end

function Base.getindex(b::TrivialParamBlock{A},i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.data
  else
    testvalue(A)
  end
end

function Base.setindex!(b::TrivialParamBlock{A},v,i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.data = v
  end
end

get_param_data(b::TrivialParamBlock) = Fill(b.data,b.plength)
param_length(b::TrivialParamBlock) = b.plength
param_getindex(b::TrivialParamBlock,i::Integer) = b.data
param_setindex!(b::TrivialParamBlock,v,i::Integer) = copyto!(b.data,v)

function get_param_entry!(v::AbstractVector,b::TrivialParamBlock,i...)
  vk = b.data[i...]
  fill!(v,vk)
end

Base.copy(a::TrivialParamBlock) = TrivialParamBlock(copy(a.data),a.plength)

Base.copyto!(a::TrivialParamBlock,b::TrivialParamBlock) = copyto!(a.data,b.data)

function Base.:≈(a::TrivialParamBlock,b::TrivialParamBlock)
  if size(a) != size(b)
    return false
  end
  a.data ≈ b.data
end

function Base.:(==)(a::TrivialParamBlock,b::TrivialParamBlock)
  if size(a) != size(b)
    return false
  end
  a.data == b.data
end

function Arrays.testvalue(a::TrivialParamBlock{A}) where A
  TrivialParamBlock(testvalue(A),param_length(a))
end

# this one misses the param length
function Arrays.testvalue(::Type{TrivialParamBlock{A}}) where A
  TrivialParamBlock(testvalue(A),1)
end

function Arrays.CachedArray(a::TrivialParamBlock)
  TrivialParamBlock(CachedArray(a.data),a.plength)
end

function Fields.unwrap_cached_array(a::TrivialParamBlock)
  TrivialParamBlock(Fields.unwrap_cached_array(a.data),a.plength)
end

###################### trivial case ######################

function Arrays.return_cache(f::GenericParamBlock,x)
  fi = testitem(f)
  li = return_cache(fi,x)
  fix = evaluate!(li,fi,x)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(f.data[i],x)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,f::GenericParamBlock,x)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],f.data[i],x)
  end
  g
end

function Fields.linear_combination(u::GenericParamBlock,f::GenericParamBlock)
  @check size(u) == size(f)
  fi = testitem(f)
  ui = testitem(u)
  ufi = linear_combination(ui,fi)
  g = Vector{typeof(ufi)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = linear_combination(u.data[i],f.data[i])
  end
  GenericParamBlock(g)
end

function Fields.linear_combination(u::GenericParamBlock,f::AbstractVector{<:Field})
  ufi = linear_combination(testitem(u),f)
  g = Vector{typeof(ufi)}(undef,param_length(u))
  @inbounds for i in param_eachindex(u)
    g[i] = linear_combination(param_getindex(u,i),f)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::LinearCombinationMap,u::GenericParamBlock,fx::AbstractArray)
  ui = testitem(u)
  li = return_cache(k,ui,fx)
  ufxi = evaluate!(li,k,ui,fx)
  l = Vector{typeof(li)}(undef,length(u.data))
  g = Vector{typeof(ufxi)}(undef,length(u.data))
  for i in eachindex(u.data)
    l[i] = return_cache(k,u.data[i],fx)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::GenericParamBlock,fx::AbstractArray)
  g,l = cache
  for i in eachindex(u.data)
    g.data[i] = evaluate!(l[i],k,u.data[i],fx)
  end
  g
end

function Arrays.return_cache(k::LinearCombinationMap,u::GenericParamBlock,fx::GenericParamBlock)
  fxi = testitem(fx)
  ui = testitem(u)
  li = return_cache(k,ui,fxi)
  ufxi = evaluate!(li,k,ui,fxi)
  l = Vector{typeof(li)}(undef,length(fx.data))
  g = Vector{typeof(ufxi)}(undef,length(fx.data))
  for i in eachindex(fx.data)
    l[i] = return_cache(k,u.data[i],fx.data[i])
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::GenericParamBlock,fx::GenericParamBlock)
  g,l = cache
  for i in eachindex(fx.data)
    g.data[i] = evaluate!(l[i],k,u.data[i],fx.data[i])
  end
  g
end

function Base.transpose(f::GenericParamBlock)
  fi = testitem(f)
  fit = transpose(fi)
  g = Vector{typeof(fit)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = transpose(f.data[i])
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Fields.TransposeMap,f::GenericParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i])
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Fields.TransposeMap,f::GenericParamBlock)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i])
  end
  g
end

function Fields.integrate(f::GenericParamBlock,args...)
  fi = testitem(f)
  intfi = integrate(fi,args...)
  g = Vector{typeof(intfi)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = integrate(f.data[i],args...)
  end
  GenericParamBlock(g)
end

function Arrays.return_value(k::IntegrationMap,fx::GenericParamBlock,args...)
  fxi = testitem(fx)
  ufxi = return_value(k,fxi,args...)
  g = Vector{typeof(ufxi)}(undef,length(fx.data))
  for i in eachindex(fx.data)
    g[i] = return_value(k,fx.data[i],args...)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::IntegrationMap,fx::GenericParamBlock,args...)
  fxi = testitem(fx)
  li = return_cache(k,fxi,args...)
  ufxi = evaluate!(li,k,fxi,args...)
  l = Vector{typeof(li)}(undef,length(fx.data))
  g = Vector{typeof(ufxi)}(undef,length(fx.data))
  for i in eachindex(fx.data)
    l[i] = return_cache(k,fx.data[i],args...)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx::GenericParamBlock,args...)
  g,l = cache
  for i in eachindex(fx.data)
    g.data[i] = evaluate!(l[i],k,fx.data[i],args...)
  end
  g
end

function Arrays.return_value(k::Broadcasting,f::GenericParamBlock)
  fi = testitem(f)
  fix = return_value(k,fi)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(k,f.data[i])
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting,f::GenericParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i])
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting,f::GenericParamBlock)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i])
  end
  g
end

function Arrays.return_value(k::Broadcasting{typeof(∘)},f::GenericParamBlock,h::Field)
  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(k,f.data[i],h)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::GenericParamBlock,h::Field)
  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = Vector{typeof(li)}(undef,length(f.data))
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i],h)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::GenericParamBlock,h::Field)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i],h)
  end
  g
end

for F in (:Function,:Operation)
  for T in (:Field,:AbstractArray)
    @eval begin
      function Arrays.return_value(k::Broadcasting{<:$F},f::GenericParamBlock,h::$T)
        fi = testitem(f)
        fix = return_value(k,fi,h)
        g = Vector{typeof(fix)}(undef,length(f.data))
        for i in eachindex(f.data)
          g[i] = return_value(k,f.data[i],h)
        end
        GenericParamBlock(g)
      end

      function Arrays.return_cache(k::Broadcasting{<:$F},f::GenericParamBlock,h::$T)
        fi = testitem(f)
        li = return_cache(k,fi,h)
        fix = evaluate!(li,k,fi,h)
        l = Vector{typeof(li)}(undef,length(f.data))
        g = Vector{typeof(fix)}(undef,length(f.data))
        for i in eachindex(f.data)
          l[i] = return_cache(k,f.data[i],h)
        end
        GenericParamBlock(g),l
      end

      function Arrays.evaluate!(cache,k::Broadcasting{<:$F},f::GenericParamBlock,h::$T)
        g,l = cache
        for i in eachindex(f.data)
          g.data[i] = evaluate!(l[i],k,f.data[i],h)
        end
        g
      end

      function Arrays.return_value(k::Broadcasting{<:$F},h::$T,f::GenericParamBlock)
        fi = testitem(f)
        fix = return_value(k,h,fi)
        g = Vector{typeof(fix)}(undef,length(f.data))
        for i in eachindex(f.data)
          g[i] = return_value(k,h,f.data[i])
        end
        GenericParamBlock(g)
      end

      function Arrays.return_cache(k::Broadcasting{<:$F},h::$T,f::GenericParamBlock)
        fi = testitem(f)
        li = return_cache(k,h,fi)
        fix = evaluate!(li,k,h,fi)
        l = Vector{typeof(li)}(undef,length(f.data))
        g = Vector{typeof(fix)}(undef,length(f.data))
        for i in eachindex(f.data)
          l[i] = return_cache(k,h,f.data[i])
        end
        GenericParamBlock(g),l
      end

      function Arrays.evaluate!(cache,k::Broadcasting{<:$F},h::$T,f::GenericParamBlock)
        g,l = cache
        for i in eachindex(f.data)
          g.data[i] = evaluate!(l[i],k,h,f.data[i])
        end
        g
      end
    end
  end
end

function Arrays.return_value(k::Broadcasting{<:Operation},h::GenericParamBlock,f::GenericParamBlock)
  @check param_length(h) == param_length(f)
  hi = testitem(h)
  fi = testitem(f)
  fix = return_value(k,hi,fi)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(k,h.data[i],f.data[i])
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},h::GenericParamBlock,f::GenericParamBlock)
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
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},h::GenericParamBlock,f::GenericParamBlock)
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

function Arrays.return_value(k::Broadcasting{<:ParamOperation},f::GenericParamBlock,h::Field)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  fix = return_value(ki,fi,h)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(param_getindex(k,i),f.data[i],h)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},f::GenericParamBlock,h::Field)
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
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ParamOperation},f::GenericParamBlock,h::Field)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],param_getindex(k,i),f.data[i],h)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ParamOperation},h::Field,f::GenericParamBlock)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  fix = return_value(ki,h,fi)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(param_getindex(k,i),h,f.data[i])
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},h::Field,f::GenericParamBlock)
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
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ParamOperation},h::Field,f::GenericParamBlock)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],param_getindex(k,i),h,f.data[i])
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ParamOperation},h::GenericParamBlock,f::GenericParamBlock)
  @check param_length(k) == param_length(h) == param_length(f)
  ki = testitem(k)
  hi = testitem(h)
  fi = testitem(f)
  fix = return_value(ki,hi,fi)
  g = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    g[i] = return_value(param_getindex(k,i),h.data[i],f.data[i])
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},h::GenericParamBlock,f::GenericParamBlock)
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
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ParamOperation},h::GenericParamBlock,f::GenericParamBlock)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],param_getindex(k,i),h.data[i],f.data[i])
  end
  g
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::GenericParamBlock,g::AbstractArray)
  fi = testitem(f)
  fix = return_value(k,fi,g)
  h = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    h[i] = return_value(k,f.data[i],g)
  end
  GenericParamBlock(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::GenericParamBlock,g::AbstractArray)
  fi = testitem(f)
  li = return_cache(k,fi,g)
  fix = evaluate!(li,k,fi,g)
  l = Vector{typeof(li)}(undef,length(f.data))
  h = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i],g)
  end
  GenericParamBlock(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::GenericParamBlock,g::AbstractArray)
  h,l = cache
  for i in eachindex(f.data)
    h.data[i] = evaluate!(l[i],k,f.data[i],g)
  end
  h
end

function Arrays.return_value(k::BroadcastingFieldOpMap,g::AbstractArray,f::GenericParamBlock)
  fi = testitem(f)
  fix = return_value(k,g,fi)
  h = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    h[i] = return_value(k,g,f.data[i])
  end
  GenericParamBlock(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,g::AbstractArray,f::GenericParamBlock)
  fi = testitem(f)
  li = return_cache(k,g,fi)
  fix = evaluate!(li,k,g,fi)
  l = Vector{typeof(li)}(undef,length(f.data))
  h = Vector{typeof(fix)}(undef,length(f.data))
  for i in eachindex(f.data)
    l[i] = return_cache(k,g,f.data[i])
  end
  GenericParamBlock(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,g::AbstractArray,f::GenericParamBlock)
  h,l = cache
  for i in eachindex(f.data)
    h.data[i] = evaluate!(l[i],k,g,f.data[i])
  end
  h
end

for op in (:+,:-,:*)
  @eval begin

    function Arrays.return_value(k::Broadcasting{typeof($op)},f::GenericParamBlock,g::GenericParamBlock)
      return_value(BroadcastingFieldOpMap($op),f,g)
    end

    function Arrays.return_cache(k::Broadcasting{typeof($op)},f::GenericParamBlock,g::GenericParamBlock)
      return_cache(BroadcastingFieldOpMap($op),f,g)
    end

    function Arrays.evaluate!(cache,k::Broadcasting{typeof($op)},f::GenericParamBlock,g::GenericParamBlock)
      evaluate!(cache,BroadcastingFieldOpMap($op),f,g)
    end

  end
end

function Arrays.return_value(k::Broadcasting{typeof(*)},f::Number,g::GenericParamBlock)
  gi = testitem(g)
  hi = return_value(k,f,gi)
  data = Vector{typeof(hi)}(undef,length(g.data))
  for i in eachindex(g.data)
    data[i] = return_value(k,f,g.data[i])
  end
  GenericParamBlock(data)
end

function Arrays.return_cache(k::Broadcasting{typeof(*)},f::Number,g::GenericParamBlock)
  gi = testitem(g)
  ci = return_cache(k,f,gi)
  hi = evaluate!(ci,k,f,gi)
  data = Vector{typeof(hi)}(undef,length(g.data))
  c = Vector{typeof(ci)}(undef,length(g.data))
  for i in eachindex(g.data)
    c[i] = return_cache(k,f,g.data[i])
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(*)},f::Number,g::GenericParamBlock)
  r,c = cache
  for i in eachindex(g.data)
    r.data[i] = evaluate!(c[i],k,f,g.data[i])
  end
  r
end

function Arrays.return_value(k::Broadcasting{typeof(*)},f::GenericParamBlock,g::Number)
  evaluate(k,f,g)
end

function Arrays.return_cache(k::Broadcasting{typeof(*)},f::GenericParamBlock,g::Number)
  return_cache(k,g,f)
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(*)},f::GenericParamBlock,g::Number)
  evaluate!(cache,k,g,f)
end

function Arrays.return_value(k::BroadcastingFieldOpMap,a::ParamBlock...)
  evaluate(k,a...)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,a::TrivialParamBlock...)
  a1 = first(a)
  @notimplementedif any(ai->param_length(ai)!=param_length(a1),a)
  ais = map(ai->ai.data,a)
  ci = return_cache(k,ais...)
  bi = evaluate!(ci,k,ais...)
  TrivialParamBlock(bi,a1.plength),ci
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::TrivialParamBlock...)
  a1 = first(a)
  @notimplementedif any(ai->param_length(ai)!=param_length(a1),a)
  r,c = cache
  ais = map(ai->ai.data,a)
  copyto!(r.data,evaluate!(c,k,ais...))
  r
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::ParamBlock,g::ParamBlock)
  @notimplementedif param_length(f) != param_length(g)
  fi = testitem(f)
  gi = testitem(g)
  hi = return_value(k,fi,gi)
  a = Vector{typeof(hi)}(undef,param_length(f))
  fill!(a,hi)
  GenericParamBlock(a)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::ParamBlock,g::ParamBlock)
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
  GenericParamBlock(a),b
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::ParamBlock,g::ParamBlock)
  @notimplementedif param_length(f) != param_length(g)
  a,b = cache
  for i in param_eachindex(f)
    a.data[i] = evaluate!(b[i],k,param_getindex(f,i),param_getindex(g,i))
  end
  a
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,a::ParamBlock...)
  a1 = first(a)
  @notimplementedif any(ai->param_length(ai)!=param_length(a1),a)
  ais = map(testitem,a)
  ci = return_cache(k,ais...)
  bi = evaluate!(ci,k,ais...)
  c = Vector{typeof(ci)}(undef,param_length(a1))
  data = Vector{typeof(bi)}(undef,param_length(a1))
  for i in param_eachindex(a1)
    _ais = map(ai->param_getindex(ai,i),a)
    c[i] = return_cache(k,_ais...)
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::ParamBlock...)
  a1 = first(a)
  @notimplementedif any(ai->param_length(ai)!=param_length(a1),a)
  r,c = cache
  for i in param_eachindex(a1)
    ais = map(ai->param_getindex(ai,i),a)
    r.data[i] = evaluate!(c[i],k,ais...)
  end
  r
end

function Arrays.return_value(
  k::BroadcastingFieldOpMap,a::Union{ParamBlock,AbstractArray}...)
  evaluate(k,a...)
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,a::Union{ParamBlock,AbstractArray}...)

  return_cache(k,lazy_parameterize(a...)...)
end

function Arrays.evaluate!(
  cache,k::BroadcastingFieldOpMap,a::Union{ParamBlock,AbstractArray}...)

  evaluate!(cache,k,lazy_parameterize(a...)...)
end

for op in (:+,:-)
  @eval begin
    function $op(a::ParamBlock,b::ParamBlock)
      BroadcastingFieldOpMap($op)(a,b)
    end

    function $op(a::TrivialParamBlock,b::TrivialParamBlock)
      @check size(a) == size(b)
      c = $op(a.data,b.data)
      TrivialParamBlock(c,a.plength)
    end
  end
end

function Base.:*(a::Number,b::GenericParamBlock)
  bi = testitem(b)
  ci = a*bi
  data = Vector{typeof(ci)}(undef,length(b.data))
  for i in eachindex(b.data)
    data[i] = a*b.data[i]
  end
  GenericParamBlock(data)
end

function Base.:*(a::Number,b::TrivialParamBlock)
  TrivialParamBlock(a*b.data,b.plength)
end

function Base.:*(a::ParamBlock,b::Number)
  b*a
end

function Base.:*(a::TrivialParamBlock,b::TrivialParamBlock)
  @check param_length(a) == param_length(b)
  c = a.data*b.data
  TrivialParamBlock(c,a.plength)
end

function LinearAlgebra.mul!(c::TrivialParamBlock,a::TrivialParamBlock,b::TrivialParamBlock)
  mul!(c.data,a.data,b.data,1,0)
end

function LinearAlgebra.rmul!(a::TrivialParamBlock,β)
  rmul!(a.data,β)
end

function Arrays.return_value(::typeof(*),a::TrivialParamBlock,b::TrivialParamBlock)
  evaluate(*,a,b)
end

function Arrays.return_cache(::typeof(*),a::TrivialParamBlock,b::TrivialParamBlock)
  CachedArray(a*b)
end

function Arrays.evaluate!(cache,::typeof(*),a::TrivialParamBlock,b::TrivialParamBlock)
  Fields._setsize_mul!(cache,a.data,b.data)
  r = cache.data
  mul!(r,a.data,b.data)
  r
end

function Base.:*(a::ParamBlock,b::ParamBlock)
  @check param_length(a) == param_length(b)
  ai = testitem(a)
  bi = testitem(b)
  ri = ai*bi
  data = Vector{typeof(ri)}(undef,param_length(a))
  data[1] = ri
  for i in 2:param_length(a)
    data[i] = param_getindex(a,i)*param_getindex(b,i)
  end
  GenericParamBlock(data)
end

function Arrays.return_value(::typeof(*),a::ParamBlock,b::ParamBlock)
  @check param_length(a) == param_length(b)
  ai = testitem(a)
  bi = testitem(b)
  ri = return_value(*,ai,bi)
  data = Vector{typeof(ri)}(undef,param_length(a))
  GenericParamBlock(data)
end

function LinearAlgebra.rmul!(a::GenericParamBlock,β)
  for i in eachindex(a.data)
    rmul!(a.data[i],β)
  end
end

function Fields._zero_entries!(a::GenericParamBlock)
  for i in eachindex(a.data)
    Fields._zero_entries!(a.data[i])
  end
end

function LinearAlgebra.mul!(c::ParamBlock,a::ParamBlock,b::ParamBlock)
  Fields._zero_entries!(c)
  mul!(c,a,b,1,0)
end

function LinearAlgebra.mul!(
  c::ParamBlock,
  a::ParamBlock,
  b::ParamBlock,
  α::Number,β::Number)

  for i in eachindex(c.data)
    mul!(param_getindex(c,i),param_getindex(a,i),param_getindex(b,i),α,β)
  end
end

function Fields._setsize_mul!(c,a::ParamBlock,b::ParamBlock)
  for i in eachindex(c.data)
    Fields._setsize_mul!(param_getindex(c,i),param_getindex(a,i),param_getindex(b,i))
  end
end

function Arrays.return_cache(::typeof(*),a::ParamBlock,b::ParamBlock)
  c1 = CachedArray(a*b)
  c2 = return_cache(Fields.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::ParamBlock,b::ParamBlock)
  c1,c2 = cache
  Fields._setsize_mul!(c1,a,b)
  c = evaluate!(c2,Fields.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Fields._setsize_as!(d,a::GenericParamBlock)
  for i in eachindex(a.data)
    Fields._setsize_mul!(param_getindex(d,i),param_getindex(a,i))
  end
  d
end

function Arrays.return_value(k::MulAddMap,a::ParamBlock,b::ParamBlock,c::ParamBlock)
  x = return_value(*,a,b)
  return_value(+,x,c)
end

function Arrays.return_cache(k::MulAddMap,a::ParamBlock,b::ParamBlock,c::ParamBlock)
  c1 = CachedArray(a*b+c)
  c2 = return_cache(unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,k::MulAddMap,a::ParamBlock,b::ParamBlock,c::ParamBlock)
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

function Arrays.return_cache(k::Arrays.ConfigMap{typeof(ForwardDiff.gradient)},x::GenericParamBlock)
  xi = testitem(x)
  fi = return_cache(k,xi)
  data = Vector{typeof(fi)}(undef,length(x.data))
  for i in eachindex(x.data)
    data[i] = return_cache(k,x.data[i])
  end
  GenericParamBlock(data)
end

function Arrays.return_cache(k::Arrays.ConfigMap{typeof(ForwardDiff.jacobian)},x::GenericParamBlock)
  xi = testitem(x)
  fi = return_cache(k,xi)
  data = Vector{typeof(fi)}(undef,length(x.data))
  for i in eachindex(x.data)
    data[i] = return_cache(k,x.data[i])
  end
  GenericParamBlock(data)
end

function Arrays.return_cache(k::Arrays.DualizeMap,x::GenericParamBlock)
  cfg = return_cache(Arrays.ConfigMap(k.f),x)
  xi = testitem(x)
  cfgi = testitem(cfg)
  xidual = evaluate!(cfgi,k,xi)
  data = Vector{typeof(xidual)}(undef,length(x.data))
  cfg,GenericParamBlock(data)
end

function Arrays.evaluate!(cache,k::Arrays.DualizeMap,x::GenericParamBlock)
  cfg,xdual = cache
  for i in eachindex(x.data)
    xdual.data[i] = evaluate!(cfg.data[i],k,x.data[i])
  end
  xdual
end

function Arrays.return_cache(k::Arrays.AutoDiffMap,ydual::GenericParamBlock,x,cfg::GenericParamBlock)
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
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,k::Arrays.AutoDiffMap,ydual::GenericParamBlock,x,cfg::GenericParamBlock)
  r,c = cache
  for i in eachindex(ydual.data)
    r.data[i] = evaluate!(c[i],k,ydual.data[i],x.data[i],cfg.data[i])
  end
  r
end

function Arrays.return_cache(k::CellData.ZeroVectorMap,a::TrivialParamBlock)
  c = return_cache(k,a.data)
  data = evaluate!(ci,k,a.data)
  TrivialParamBlock(data,v.plength),c
end

function Arrays.evaluate!(cache,k::CellData.ZeroVectorMap,a::TrivialParamBlock)
  r,c = cache
  copyto!(r.data,evaluate!(c[i],k,a.data))
  r
end

function Arrays.return_cache(k::CellData.ZeroVectorMap,a::GenericParamBlock)
  ai = testitem(a)
  ci = return_cache(k,ai)
  vi = evaluate!(ci,k,ai)
  c = Vector{typeof(ci)}(undef,param_length(a))
  data = Vector{typeof(vi)}(undef,param_length(a))
  for i in 1:param_length(a)
    c[i] = return_cache(k,param_getindex(a,i))
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,k::CellData.ZeroVectorMap,a::GenericParamBlock)
  r,c = cache
  for i in eachindex(ydual.data)
    r.data[i] = evaluate!(c[i],k,a.data[i])
  end
  r
end

# cell datas

function Geometry._cache_compress(data::ParamBlock)
  c1 = CachedArray(deepcopy(data))
  c2 = return_cache(Fields.unwrap_cached_array,c1)
  c1,c2
end

function Geometry._setempty_compress!(a::TrivialParamBlock)
  Geometry._setempty_compress!(a.data)
end

function Geometry._setempty_compress!(a::GenericParamBlock)
  for i in eachindex(a.data)
    Geometry._setempty_compress!(a.data[i])
  end
end

function Geometry._uncached_compress!(c1::ParamBlock,c2)
  evaluate!(c2,Fields.unwrap_cached_array,c1)
end

function Geometry._setsize_compress!(a::TrivialParamBlock,b::TrivialParamBlock)
  Geometry._setsize_compress!(a.data,b.data)
end

function Geometry._setsize_compress!(a::ParamBlock,b::ParamBlock)
  @check size(a) == size(b)
  for i in param_eachindex(a)
    Geometry._setsize_compress!(param_getindex(a,i),param_getindex(b,i))
  end
end

function Geometry._copyto_compress!(a::TrivialParamBlock,b::TrivialParamBlock)
  Geometry._copyto_compress!(a.data,b.data)
end

function Geometry._copyto_compress!(a::ParamBlock,b::ParamBlock)
  @check size(a) == size(b)
  for i in param_eachindex(a)
    Geometry._copyto_compress!(param_getindex(a,i),param_getindex(b,i))
  end
end

function Geometry._addto_compress!(a::TrivialParamBlock,b::TrivialParamBlock)
  Geometry._addto_compress!(a.data,b.data)
end

function Geometry._addto_compress!(a::ParamBlock,b::ParamBlock)
  @check size(a) == size(b)
  for i in param_eachindex(a)
    Geometry._addto_compress!(param_getindex(a,i),param_getindex(b,i))
  end
end

function Geometry._similar_empty(val::TrivialParamBlock)
  TrivialParamBlock(Geometry._similar_empty(val.data),val.plength)
end

function Geometry._similar_empty(val::GenericParamBlock)
  a = deepcopy(val)
  for i in eachindex(a)
    a.data[i] = Geometry._similar_empty(a.data[i])
  end
  a
end

function Geometry.pos_neg_data(
  ipos_to_val::AbstractArray{<:ParamBlock},i_to_iposneg::PosNegPartition)
  nineg = length(i_to_iposneg.ineg_to_i)
  val = testitem(ipos_to_val)
  void = Geometry._similar_empty(val)
  ineg_to_val = Fill(void,nineg)
  ipos_to_val,ineg_to_val
end

# reference FEs

function Arrays.return_cache(b::LagrangianDofBasis,f::GenericParamBlock)
  fi = testitem(f)
  ci = return_cache(b,fi)
  ri = evaluate!(ci,b,fi)
  c = Vector{typeof(ci)}(undef,length(f.data))
  data = Vector{typeof(ri)}(undef,length(f.data))
  for i in eachindex(f.data)
    c[i] = return_cache(b,f.data[i])
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,b::LagrangianDofBasis,f::GenericParamBlock)
  r,c = cache
  for i in eachindex(f.data)
    r.data[i] = evaluate!(c[i],b,f.data[i])
  end
  r
end

# block map interface

function Arrays.return_value(k::BroadcastingFieldOpMap,f::ArrayBlock{A,N},h::ParamBlock) where {A,N}
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

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::ArrayBlock{A,N},h::ParamBlock) where {A,N}
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

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::ArrayBlock,h::ParamBlock)
  g,l = cache
  @check g.touched == f.touched
  for i in eachindex(f.array)
    if f.touched[i]
      g.array[i] = evaluate!(l[i],k,f.array[i],h)
    end
  end
  g
end

function Arrays.return_value(k::BroadcastingFieldOpMap,h::ParamBlock,f::ArrayBlock{A,N}) where {A,N}
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

function Arrays.return_cache(k::BroadcastingFieldOpMap,h::ParamBlock,f::ArrayBlock{A,N}) where {A,N}
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

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,h::ParamBlock,f::ArrayBlock)
  g,l = cache
  @check g.touched == f.touched
  for i in eachindex(f.array)
    if f.touched[i]
      g.array[i] = evaluate!(l[i],k,h,f.array[i])
    end
  end
  g
end

for S in (:ParamBlock,:AbstractArray)
  for T in (:ParamBlock,:AbstractArray)
    (S == :AbstractArray && T == :AbstractArray) && continue
    @eval begin
      function Arrays.return_cache(
        k::BroadcastingFieldOpMap,
        f::ArrayBlock{<:$S,N},
        g::ArrayBlock{<:$T,N}
        ) where N

        @notimplementedif size(f) != size(g)
        fi,gi = _test_item_values(f,g)
        ci = return_cache(k,fi,gi)
        hi = evaluate!(ci,k,fi,gi)
        m = Fields.ZeroBlockMap()
        a = Array{typeof(hi),N}(undef,size(f.array))
        b = Array{typeof(ci),N}(undef,size(f.array))
        zf = Array{typeof(return_cache(m,fi,gi))}(undef,size(f.array))
        zg = Array{typeof(return_cache(m,gi,fi))}(undef,size(f.array))
        t = map(|,f.touched,g.touched)
        for i in eachindex(f.array)
          if f.touched[i] && g.touched[i]
            b[i] = return_cache(k,f.array[i],g.array[i])
          elseif f.touched[i]
            _fi = f.array[i]
            zg[i] = return_cache(m,gi,_fi)
            _gi = evaluate!(zg[i],m,gi,_fi)
            b[i] = return_cache(k,_fi,_gi)
          elseif g.touched[i]
            _gi = g.array[i]
            zf[i] = return_cache(m,fi,_gi)
            _fi = evaluate!(zf[i],m,fi,_gi)
            b[i] = return_cache(k,_fi,_gi)
          end
        end
        ArrayBlock(a,t),b,zf,zg
      end

      function Arrays.return_cache(
        k::BroadcastingFieldOpMap,
        f::ArrayBlock{<:$S,1},
        g::ArrayBlock{<:$T,2}
        )

        fi,gi = _test_item_values(f,g)
        ci = return_cache(k,fi,gi)
        hi = evaluate!(ci,k,fi,gi)
        @check size(g.array,1) == 1 || size(g.array,2) == 0
        s = (size(f.array,1),size(g.array,2))
        a = Array{typeof(hi),2}(undef,s)
        b = Array{typeof(ci),2}(undef,s)
        t = fill(false,s)
        for j in 1:s[2]
          for i in 1:s[1]
            if f.touched[i] && g.touched[1,j]
              t[i,j] = true
              b[i,j] = return_cache(k,f.array[i],g.array[1,j])
            end
          end
        end
        ArrayBlock(a,t),b
      end

      function Arrays.return_cache(
        k::BroadcastingFieldOpMap,
        f::ArrayBlock{<:$S,2},
        g::ArrayBlock{<:$T,1}
        )

        fi,gi = _test_item_values(f,g)
        ci = return_cache(k,fi,gi)
        hi = evaluate!(ci,k,fi,gi)
        @check size(f.array,1) == 1 || size(f.array,2) == 0
        s = (size(g.array,1),size(f.array,2))
        a = Array{typeof(hi),2}(undef,s)
        b = Array{typeof(ci),2}(undef,s)
        t = fill(false,s)
        for j in 1:s[2]
          for i in 1:s[1]
            if f.touched[1,j] && g.touched[i]
              t[i,j] = true
              b[i,j] = return_cache(k,f.array[1,j],g.array[i])
            end
          end
        end
        ArrayBlock(a,t),b
      end
    end
  end
end

function Arrays.return_value(k::BroadcastingFieldOpMap,a::Union{ArrayBlock,ParamBlock}...)
  evaluate(k,a...)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,a::Union{ArrayBlock,ParamBlock}...)
  function _replace_nz_blocks(a::ArrayBlock,bi::ParamBlock)
    N = ndims(a.array)
    array = Array{typeof(bi),N}(undef,size(a))
    for i in eachindex(a.array)
      if a.touched[i]
        array[i] = bi
      end
    end
    ArrayBlock(array,a.touched)
  end

  function _replace_nz_blocks(a::ArrayBlock,bi::ArrayBlock)
    bi
  end

  inds = findall(ai->isa(ai,ArrayBlock),a)
  @notimplementedif length(inds) == 0
  a1 = a[inds[1]]
  b = map(ai->_replace_nz_blocks(a1,ai),a)
  c = return_cache(k,b...)
  c,b
end

for A in (:ArrayBlock,:ParamBlock)
  for B in (:ArrayBlock,:ParamBlock)
    for C in (:ArrayBlock,:ParamBlock)
      if !(A == B == C)
        @eval begin
          function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::$A,b::$B,c::$C)
            eval_cache,replace_cache = cache
            cachea,cacheb,cachec = replace_cache

            _replace_nz_blocks!(cachea,a)
            _replace_nz_blocks!(cacheb,b)
            _replace_nz_blocks!(cachec,c)

            evaluate!(eval_cache,k,cachea,cacheb,cachec)
          end
        end
      end
      for D in (:ArrayBlock,:ParamBlock)
        if !(A == B == C == D)
          @eval begin
            function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::$A,b::$B,c::$C,d::$D)
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

function Arrays.return_cache(k::Fields.ZeroBlockMap,h::ParamBlock,f::ParamBlock)
  @check param_length(h) == param_length(f)
  hi = testitem(h)
  fi = testitem(f)
  ci = return_cache(k,hi,fi)
  ri = evaluate!(ci,k,hi,fi)
  c = Vector{typeof(ci)}(undef,param_length(f))
  data = Vector{typeof(ri)}(undef,param_length(f))
  for i in param_eachindex(f)
    c[i] = return_cache(k,param_getindex(h,i),param_getindex(f,i))
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache::Tuple,k::Fields.ZeroBlockMap,h::ParamBlock,f::ParamBlock)
  g,l = cache
  for i in eachindex(g.data)
    g.data[i] = evaluate!(l[i],k,param_getindex(h,i),param_getindex(f,i))
  end
  g
end

for T in (:AbstractArray,:Nothing)
  @eval begin
    function Arrays.return_cache(k::Fields.ZeroBlockMap,h::ParamBlock,f::$T)
      return_cache(k,h,lazy_parameterize(f,param_length(h)))
    end
    function Arrays.return_cache(k::Fields.ZeroBlockMap,h::$T,f::ParamBlock)
      return_cache(k,lazy_parameterize(h,param_length(f)),f)
    end
    function Arrays.evaluate!(cache::Tuple,k::Fields.ZeroBlockMap,h::ParamBlock,f::$T)
      evaluate!(cache,k,h,lazy_parameterize(f,param_length(h)))
    end
    function Arrays.evaluate!(cache::Tuple,k::Fields.ZeroBlockMap,h::$T,f::ParamBlock)
      evaluate!(cache,k,lazy_parameterize(h,param_length(f)),f)
    end
    function Arrays.evaluate!(cache::Tuple,k::Fields.ZeroBlockMap,h::$T,f::AbstractArray)
      plength = param_length(cache[1])
      evaluate!(cache,k,lazy_parameterize(h,plength),lazy_parameterize(f,plength))
    end
  end
end

# constructors

function lazy_parameterize(a::ParamBlock,plength::Integer=param_length(a))
  @check param_length(a) == plength
  a
end

function lazy_parameterize(a::Union{AbstractArray{<:Number},Nothing,Field,AbstractArray{<:Field}},plength::Integer)
  TrivialParamBlock(a,plength)
end

function local_parameterize(a::Union{AbstractArray{<:Number},Nothing,Field,AbstractArray{<:Field}},plength::Integer)
  data = Vector{typeof(a)}(undef,plength)
  @inbounds for i in 1:plength
    data[i] = copy(a)
  end
  GenericParamBlock(data)
end

function local_parameterize(a::AbstractArray{<:AbstractArray},plength::Integer)
  @check length(a) == plength
  GenericParamBlock(a)
end

function local_parameterize(a::ParamBlock,plength::Integer)
  @check param_length(a) == plength
  a
end

function Fields.GenericField(f::AbstractParamFunction)
  GenericParamBlock(map(i -> GenericField(f[i]),1:length(f)))
end

# need to correct this function

for T in (:ParamBlock,:(ArrayBlock{<:ParamBlock}))
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

function _replace_nz_blocks!(cache::ArrayBlock,val::ParamBlock)
  for i in eachindex(cache.array)
    if cache.touched[i]
      cache.array[i] = val
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

function _test_values(h::ParamBlock,f::ParamBlock)
  @check param_length(h) == param_length(f)
  hi = testvalue(h)
  fi = testvalue(f)
  return hi,fi
end

function _test_values(h::ParamBlock,_f)
  f = lazy_parameterize(_f,param_length(h))
  _test_values(h,f)
end

function _test_values(_h,f::ParamBlock)
  h = lazy_parameterize(_h,param_length(f))
  _test_values(h,f)
end

function _test_item_values(h::ArrayBlock,f::ArrayBlock)
  _test_values(testitem(h),testitem(f))
end
