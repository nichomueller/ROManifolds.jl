abstract type ParamUnit{A,N} end

Base.size(b::ParamUnit{A,N}) where {A,N} = tfill(param_length(b),Val{N}())
Base.length(b::ParamUnit{A,N}) where {A,N} = param_length(b)^N
Base.eltype(::Type{<:ParamUnit{A}}) where A = A
Base.eltype(b::ParamUnit{A}) where A = A
Base.ndims(b::ParamUnit{A,N}) where {A,N} = N
Base.ndims(::Type{ParamUnit{A,N}}) where {A,N} = N

function Base.:≈(a::AbstractArray{<:ParamUnit},b::AbstractArray{<:ParamUnit})
  all(z->z[1]≈z[2],zip(a,b))
end

struct GenericParamUnit{A,N,S} <: ParamUnit{A,N}
  array::Vector{A}
  innersize::S
  function GenericParamUnit(array::Vector{A},innersize::S) where {A,N,S<:NTuple{N}}
    @check all(ai->size(ai)==innersize,array)
    new{A,N,S}(array,innersize)
  end
end

function GenericParamUnit(array::AbstractVector)
  innersize = size(first(array))
  @check all(ai->size(ai)==innersize,array)
  GenericParamUnit(innersize,array)
end

function Base.getindex(b::GenericParamUnit{A},i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.array[iblock]
  else
    testvalue(A)
  end
end

function Base.setindex!(b::GenericParamUnit{A},v,i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.array[iblock] = v
  end
end

param_length(b::GenericParamUnit) = length(b.array)
param_getindex(b::GenericParamUnit,i) = b.array[i]
param_setindex!(b::GenericParamUnit,v,i) = (b.array[i]=v)

Base.copy(a::GenericParamUnit) = GenericParamUnit(copy(a.array),a.innersize)

function _size_check(a::GenericParamUnit,b::GenericParamUnit)
  @check a.innersize == b.innersize && length(a.array) == length(b.array)
end

function Base.copyto!(a::GenericParamUnit,b::GenericParamUnit)
  _size_check(a,b)
  for i in eachindex(a.array)
    fill!(a.array[i],zero(eltype(a.array[i])))
    copyto!(a.array[i],b.array[i])
  end
  a
end

function Base.:≈(a::GenericParamUnit,b::GenericParamUnit)
  if size(a) != size(b)
    return false
  end
  for i in eachindex(a.array)
    if !(a.array[i] ≈ b.array[i])
      return false
    end
  end
  true
end

function Base.:(==)(a::GenericParamUnit,b::GenericParamUnit)
  if size(a) != size(b)
    return false
  end
  for i in eachindex(a.array)
    if a.array[i] != b.array[i]
      return false
    end
  end
  true
end

function Arrays.testitem(a::GenericParamUnit{A}) where A
  @notimplementedif !isconcretetype(A)
  if length(i) != 0
    testitem(a.array)
  else
    testvalue(A)
  end
end

function Arrays.testvalue(::Type{GenericParamUnit{A,N}}) where {A,N}
  array = Vector{A}(undef,0)
  innersize = tfill(0,Val{N}())
  GenericParamUnit(array,innersize)
end

function Arrays.CachedArray(a::GenericParamUnit)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci),ndims(a)}(undef,length(a))
  for i in eachindex(a.array)
    array[i] = CachedArray(a.array[i])
  end
  GenericParamUnit(array,a.innersize)
end

function Fields.unwrap_cached_array(a::GenericParamUnit)
  cache = return_cache(Fields.unwrap_cached_array,a)
  evaluate!(cache,Fields.unwrap_cached_array,a)
end

function Arrays.return_cache(::typeof(Fields.unwrap_cached_array),a::GenericParamUnit)
  ai = testitem(a)
  ci = return_cache(Fields.unwrap_cached_array,ai)
  ri = evaluate!(ci,Fields.unwrap_cached_array,ai)
  c = Vector{typeof(ci),ndims(a)}(undef,length(a))
  array = Vector{typeof(ri),ndims(a)}(undef,length(a))
  for i in eachindex(a.array)
    c[i] = return_cache(Fields.unwrap_cached_array,a.array[i])
  end
  GenericParamUnit(array,a.innersize),c
end

function Arrays.evaluate!(cache,::typeof(Fields.unwrap_cached_array),a::GenericParamUnit)
  r,c = cache
  for i in eachindex(a.array)
    r.array[i] = evaluate!(c[i],Fields.unwrap_cached_array,a.array[i])
  end
  r
end

###################### trivial case ######################

struct TrivialParamUnit{A,N} <: ParamUnit{A,N}
  array::A
  innersize::S
  plength::Int
  function TrivialParamUnit(array::A,innersize::S,plength::Int=1) where {A,N,S<:NTuple{N}}
    new{A,N,S}(array,innersize,plength)
  end
end

function TrivialParamUnit(array::Any,plength::Int=1)
  innersize = size(array)
  TrivialParamUnit(innersize,array,plength)
end

function Base.getindex(b::TrivialParamUnit{A},i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.array
  else
    testvalue(A)
  end
end

function Base.setindex!(b::TrivialParamUnit{A},v,i...) where A
  iblock = first(i)
  if all(i.==iblock)
    b.array = v
  end
end

param_length(b::TrivialParamUnit) = b.plength
param_getindex(b::TrivialParamUnit,i) = b.array
param_setindex!(b::TrivialParamUnit,v,i) = copyto!(b.array,v)

Base.copy(a::TrivialParamUnit) = TrivialParamUnit(copy(a.array),a.innersize,a.plength)

Base.copyto!(a::TrivialParamUnit,b::TrivialParamUnit) = copyto!(a.array,b.array)

function Base.:≈(a::TrivialParamUnit,b::TrivialParamUnit)
  if size(a) != size(b)
    return false
  end
  a.array ≈ b.array
end

function Base.:(==)(a::TrivialParamUnit,b::TrivialParamUnit)
  if size(a) != size(b)
    return false
  end
  a.array == b.array
end

function Arrays.testitem(a::TrivialParamUnit{A}) where A
  @notimplementedif !isconcretetype(A)
  if length(i) != 0
    a.array
  else
    testvalue(A)
  end
end

function Arrays.testvalue(::Type{TrivialParamUnit{A,N}}) where {A,N}
  array = Vector{A}(undef,0)
  innersize = tfill(0,Val{N}())
  plength = 0
  TrivialParamUnit(array,innersize,plength)
end

function Arrays.CachedArray(a::TrivialParamUnit)
  TrivialParamUnit(CachedArray(a.array),a.innersize,a.plength)
end

function Fields.unwrap_cached_array(a::TrivialParamUnit)
  TrivialParamUnit(Fields.unwrap_cached_array(a.array),a.innersize,a.plength)
end

###################### trivial case ######################

function Arrays.return_cache(f::GenericParamUnit,x)
  fi = testitem(f)
  li = return_cache(fi,x)
  fix = evaluate!(li,fi,x)
  l = Vector{typeof(li)}(undef,length(f.array))
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    l[i] = return_cache(f.array[i],x)
  end
  GenericParamUnit(g,f.innersize),l
end

function Arrays.evaluate!(cache,f::GenericParamUnit,x)
  g,l = cache
  for i in eachindex(f.array)
    g.array[i] = evaluate!(l[i],f.array[i],x)
  end
  g
end

function linear_combination(u::GenericParamUnit,f::GenericParamUnit)
  @check _size_check(u,f)
  fi = testitem(f)
  ui = testitem(u)
  ufi = linear_combination(ui,fi)
  g = Vector{typeof(ufi)}(undef,length(f.array))
  for i in eachindex(f.array)
    g[i] = linear_combination(u.array[i],f.array[i])
  end
  GenericParamUnit(g,f.innersize)
end

function Arrays.return_cache(k::LinearCombinationMap,u::GenericParamUnit,fx::GenericParamUnit)
  @check _size_check(u,f)
  fxi = testitem(fx)
  ui = testitem(u)
  li = return_cache(k,ui,fxi)
  ufxi = evaluate!(li,k,ui,fxi)
  l = Vector{typeof(li)}(undef,length(fx.array))
  g = Vector{typeof(ufxi)}(undef,length(fx.array))
  for i in eachindex(fx.array)
    l[i] = return_cache(k,u.array[i],fx.array[i])
  end
  GenericParamUnit(g,fx.innersize),l
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::GenericParamUnit,fx::GenericParamUnit)
  g,l = cache
  for i in eachindex(fx.array)
    g.array[i] = evaluate!(l[i],k,u.array[i],fx.array[i])
  end
  g
end

function Base.transpose(f::GenericParamUnit)
  fi = testitem(f)
  fit = transpose(fi)
  g = Vector{typeof(fit)}(undef,length(f.array))
  for i in eachindex(f.array)
    g[i] = transpose(f.array[i])
  end
  GenericParamUnit(g,f.innersize)
end

function Arrays.return_cache(k::TransposeMap,f::GenericParamUnit)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,length(f.array))
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    l[i] = return_cache(k,f.array[i])
  end
  GenericParamUnit(g,f.innersize),l
end

function Arrays.evaluate!(cache,k::TransposeMap,f::GenericParamUnit)
  g,l = cache
  for i in eachindex(f.array)
    g.array[i] = evaluate!(l[i],k,f.array[i])
  end
  g
end

function integrate(f::GenericParamUnit,args...)
  fi = testitem(f)
  intfi = integrate(fi,args...)
  g = Vector{typeof(intfi)}(undef,length(f.array))
  for i in eachindex(f.array)
    g[i] = integrate(f.array[i],args...)
  end
  GenericParamUnit(g,f.innersize)
end

function Arrays.return_value(k::IntegrationMap,fx::GenericParamUnit,args...)
  fxi = testitem(fx)
  ufxi = return_value(k,fxi,args...)
  g = Vector{typeof(ufxi)}(undef,length(fx.array))
  for i in eachindex(fx.array)
    g[i] = return_value(k,fx.array[i],args...)
  end
  GenericParamUnit(g,fx.innersize)
end

function Arrays.return_cache(k::IntegrationMap,fx::GenericParamUnit,args...)
  fxi = testitem(fx)
  li = return_cache(k,fxi,args...)
  ufxi = evaluate!(li,k,fxi,args...)
  l = Vector{typeof(li)}(undef,length(fx.array))
  g = Vector{typeof(ufxi)}(undef,length(fx.array))
  for i in eachindex(fx.array)
    l[i] = return_cache(k,fx.array[i],args...)
  end
  GenericParamUnit(g,fx.innersize),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx::GenericParamUnit,args...)
  g,l = cache
  for i in eachindex(fx.array)
    g.array[i] = evaluate!(l[i],k,fx.array[i],args...)
  end
  g
end

function Arrays.return_value(k::Broadcasting,f::GenericParamUnit)
  fi = testitem(f)
  fix = return_value(k,fi)
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    g[i] = return_value(k,f.array[i])
  end
  GenericParamUnit(g,f.innersize)
end

function Arrays.return_cache(k::Broadcasting,f::GenericParamUnit)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,length(f.array))
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    l[i] = return_cache(k,f.array[i])
  end
  GenericParamUnit(g,f.innersize),l
end

function Arrays.evaluate!(cache,k::Broadcasting,f::GenericParamUnit)
  g,l = cache
  for i in eachindex(f.array)
    g.array[i] = evaluate!(l[i],k,f.array[i])
  end
  g
end

function Arrays.return_value(k::Broadcasting{typeof(∘)},f::GenericParamUnit,h::Field)
  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    g[i] = return_value(k,f.array[i],h)
  end
  GenericParamUnit(g,f.innersize)
end

function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::GenericParamUnit,h::Field)
  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = Vector{typeof(li)}(undef,length(f.array))
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    l[i] = return_cache(k,f.array[i],h)
  end
  GenericParamUnit(g,f.innersize),l
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::GenericParamUnit,h::Field)
  g,l = cache
  for i in eachindex(f.array)
    g.array[i] = evaluate!(l[i],k,f.array[i],h)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:Operation},f::GenericParamUnit,h::Field)
  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    g[i] = return_value(k,f.array[i],h)
  end
  GenericParamUnit(g,f.innersize)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},f::GenericParamUnit,h::Field)
  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = Vector{typeof(li)}(undef,length(f.array))
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    l[i] = return_cache(k,f.array[i],h)
  end
  GenericParamUnit(g,f.innersize),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},f::GenericParamUnit,h::Field)
  g,l = cache
  for i in eachindex(f.array)
    g.array[i] = evaluate!(l[i],k,f.array[i],h)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:Operation},h::Field,f::GenericParamUnit)
  fi = testitem(f)
  fix = return_value(k,h,fi)
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    g[i] = return_value(k,h,f.array[i])
  end
  GenericParamUnit(g,f.innersize)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},h::Field,f::GenericParamUnit)
  fi = testitem(f)
  li = return_cache(k,h,fi)
  fix = evaluate!(li,k,h,fi)
  l = Vector{typeof(li)}(undef,length(f.array))
  g = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    l[i] = return_cache(k,h,f.array[i])
  end
  GenericParamUnit(g,f.innersize),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},h::Field,f::GenericParamUnit)
  g,l = cache
  for i in eachindex(f.array)
    g.array[i] = evaluate!(l[i],k,h,f.array[i])
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:Operation},h::GenericParamUnit,f::GenericParamUnit)
  evaluate(k,h,f)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},h::GenericParamUnit,f::GenericParamUnit{A,N}) where {A,N}
  @notimplemented
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},h::GenericParamUnit,f::GenericParamUnit)
  @notimplemented
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::GenericParamUnit,g::AbstractArray)
  fi = testitem(f)
  fix = return_value(k,fi,g)
  h = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    h[i] = return_value(k,f.array[i],g)
  end
  GenericParamUnit(h,f.innersize)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::GenericParamUnit,g::AbstractArray)
  fi = testitem(f)
  li = return_cache(k,fi,g)
  fix = evaluate!(li,k,fi,g)
  l = Vector{typeof(li)}(undef,length(f.array))
  h = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    l[i] = return_cache(k,f.array[i],g)
  end
  GenericParamUnit(h,f.innersize),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::GenericParamUnit,g::AbstractArray)
  h,l = cache
  for i in eachindex(f.array)
    h.array[i] = evaluate!(l[i],k,f.array[i],g)
  end
  h
end

function Arrays.return_value(k::BroadcastingFieldOpMap,g::AbstractArray,f::GenericParamUnit)
  fi = testitem(f)
  fix = return_value(k,g,fi)
  h = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    h[i] = return_value(k,g,f.array[i])
  end
  GenericParamUnit(h,f.innersize)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,g::AbstractArray,f::GenericParamUnit)
  fi = testitem(f)
  li = return_cache(k,g,fi)
  fix = evaluate!(li,k,g,fi)
  l = Vector{typeof(li)}(undef,length(f.array))
  h = Vector{typeof(fix)}(undef,length(f.array))
  for i in eachindex(f.array)
    l[i] = return_cache(k,g,f.array[i])
  end
  GenericParamUnit(h,f.innersize),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,g::AbstractArray,f::GenericParamUnit)
  h,l = cache
  for i in eachindex(f.array)
    h.array[i] = evaluate!(l[i],k,g,f.array[i])
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
  array = Vector{typeof(hi),ndims(g.array)}(undef,length(g.array))
  for i in eachindex(g.array)
    array[i] = return_value(k,f,g.array[i])
  end
  GenericParamUnit(array,f.innersize)
end

function Arrays.return_cache(k::Broadcasting{typeof(*)},f::Number,g::GenericParamUnit)
  gi = testitem(g)
  ci = return_cache(k,f,gi)
  hi = evaluate!(ci,k,f,gi)
  array = Vector{typeof(hi),ndims(g.array)}(undef,length(g.array))
  c = Vector{typeof(ci),ndims(g.array)}(undef,length(g.array))
  for i in eachindex(g.array)
    c[i] = return_cache(k,f,g.array[i])
  end
  GenericParamUnit(array,f.innersize),c
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(*)},f::Number,g::GenericParamUnit)
  r,c = cache
  for i in eachindex(g.array)
    r.array[i] = evaluate!(c[i],k,f,g.array[i])
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

function Arrays.return_value(k::BroadcastingFieldOpMap,f::GenericParamUnit,g::GenericParamUnit)
  evaluate(k,f,g)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::GenericParamUnit,g::GenericParamUnit)
  @notimplemented
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::GenericParamUnit,g::GenericParamUnit)
  @notimplemented
end

function Arrays.return_value(k::BroadcastingFieldOpMap,a::(ParamUnit{A,N} where A)...) where N
  evaluate(k,a...)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,a::(TrivialParamUnit{A,N} where A)...) where N
  a1 = first(a)
  @notimplementedif any(ai->size(ai)!=size(a1),a)
  ais = map(ai->testvalue(eltype(ai)),a)
  ai = return_cache(k,ais...)
  bi = evaluate!(ci,k,ais...)
  TrivialParamUnit(bi,a1.innersize,a1.plength),ai
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::(TrivialParamUnit{A,N} where A)...) where N
  a1 = first(a)
  @notimplementedif any(ai->size(ai)!=size(a1),a)
  r,c = cache
  ais = map(ai->ai.array,a)
  copyto!(r.array,evaluate!(c,k,ais...))
  r
end

function Arrays.return_value(
  k::BroadcastingFieldOpMap,f::ParamUnit{A,N},g::ParamUnit{B,N}) where {A,B,N}
  fi = testvalue(A)
  gi = testvalue(B)
  hi = return_value(k,fi,gi)
  a = Vector{typeof(hi)}(undef,param_length(f))
  fill!(a,hi)
  GenericParamUnit(a,f.innersize)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::ParamUnit{A,N},g::ParamUnit{B,N}) where {A,B,N}
  @notimplementedif size(f) != size(g)
  fi = testvalue(A)
  gi = testvalue(B)
  ci = return_cache(k,fi,gi)
  hi = evaluate!(ci,k,fi,gi)
  a = Vector{typeof(hi)}(undef,param_length(f))
  b = Vector{typeof(ci)}(undef,param_length(f))
  for i in param_eachindex(f)
    b[i] = return_cache(k,param_getindex(f,i),param_getindex(g,i))
  end
  GenericParamUnit(a,f.innersize),b
end

function Arrays.evaluate!(
  cache,k::BroadcastingFieldOpMap,f::GenericParamUnit{A,N},g::GenericParamUnit{B,N}) where {A,B,N}
  a,b = cache
  @check size(f) == size(g)
  @check size(a) == size(g)
  for i in eachindex(f.array)
    a.array[i] = evaluate!(b[i],k,param_getindex(f,i),param_getindex(g,i))
  end
  a
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,a::(ParamUnit{A,N} where A)...) where N
  a1 = first(a)
  @notimplementedif any(ai->size(ai)!=size(a1),a)
  ais = map(ai->testvalue(eltype(ai)),a)
  ci = return_cache(k,ais...)
  bi = evaluate!(ci,k,ais...)
  c = Vector{typeof(ci)}(undef,length(a1))
  array = Vector{typeof(bi)}(undef,length(a1))
  for i in eachindex(a1.array)
    _ais = map(ai->ai.array[i],a)
    c[i] = return_cache(k,_ais...)
  end
  GenericParamUnit(array,f.innersize),c
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::(ParamUnit{A,N} where A)...) where N
  a1 = first(a)
  @notimplementedif any(ai->size(ai)!=size(a1),a)
  r,c = cache
  for i in eachindex(a1.array)
    ais = map(ai->ai.array[i],a)
    r.array[i] = evaluate!(c[i],k,ais...)
  end
  r
end

function Arrays.return_value(k::BroadcastingFieldOpMap,a::GenericParamUnit...)
  evaluate(k,a...)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,a::GenericParamUnit...)
  @notimplemented
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::GenericParamUnit...)
  @notimplemented
end

function Arrays.return_value(
  k::BroadcastingFieldOpMap,a::Union{ParamUnit,AbstractArray}...)
  evaluate(k,a...)
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,a::Union{ParamUnit,AbstractArray}...)

  return_cache(k,to_param_quantities(a...)...)
end

function Arrays.evaluate!(
  cache,k::BroadcastingFieldOpMap,a::Union{GenericParamUnit,AbstractArray}...)

  evaluate!(cache,k,to_param_quantities(a...)...)
end

for op in (:+,:-)
  @eval begin
    function $op(a::ParamUnit,b::ParamUnit)
      BroadcastingFieldOpMap($op)(a,b)
    end

    function $op(a::TrivialParamUnit,b::TrivialParamUnit)
      @check size(a) == size(b)
      array = TrivialParamUnit($op(a.array,b.array))
      TrivialParamUnit(array,a.innersize,a.plength)
    end
  end
end

function Base.:*(a::Number,b::GenericParamUnit)
  bi = testitem(b)
  ci = a*bi
  array = Vector{typeof(ci)}(undef,length(b))
  for i in eachindex(b.array)
    array[i] = a*b.array[i]
  end
  GenericParamUnit(array,b.innersize)
end

function Base.:*(a::Number,b::TrivialParamUnit)
  TrivialParamUnit(a*b.array,b.innersize,b.plength)
end

function Base.:*(a::ParamUnit,b::Number)
  b*a
end

function Base.:*(a::TrivialParamUnit{A,2},b::TrivialParamUnit{B}) where {A,B}
  @check size(a.array,2) == size(b.array,1)
  @check a.plength == b.plength
  array = a.array*b.array
  TrivialParamUnit(array,size(array),a.plength)
end

function LinearAlgebra.mul!(c::TrivialParamUnit,a::TrivialParamUnit,b::TrivialParamUnit)
  mul!(c.array,a.array,b.array,1,0)
end

function LinearAlgebra.rmul!(a::TrivialParamUnit,β)
  rmul!(a.array,β)
end

function Arrays.return_value(::typeof(*),a::TrivialParamUnit,b::TrivialParamUnit)
  evaluate(*,a,b)
end

function Arrays.return_cache(::typeof(*),a::TrivialParamUnit,b::TrivialParamUnit)
  CachedArray(a*b)
end

function Arrays.evaluate!(cache,::typeof(*),a::TrivialParamUnit,b::TrivialParamUnit)
  Fields._setsize_mul!(cache,a.array,b.array)
  r = cache.array
  mul!(r,a.array,b.array)
  r
end

function Base.:*(a::ParamUnit{A,2},b::ParamUnit{B}) where {A,B}
  @check a.innersize[2] == b.innersize[1]
  ai = testvalue(A)
  bi = testvalue(B)
  ri = ai*bi
  array = Vector{typeof(ri)}(undef,param_length(a))
  for i in eachindex(b.array)
    array[i] = a.array[i]*b.array[i]
  end
  innersize = size(array[1])
  GenericParamUnit(array,innersize)
end

_prod_innersize(a::ParamUnit{A,2},b::ParamUnit{B,1}) where {A,B} = (a.innersize[1],)
_prod_innersize(a::ParamUnit{A,2},b::ParamUnit{B,2}) where {A,B} = (a.innersize[1],b.innersize[2])

function Arrays.return_value(::typeof(*),a::ParamUnit{A,2},b::ParamUnit{B}) where {A,B}
  @check param_length(a) == param_length(b)
  ai = testvalue(A)
  bi = testvalue(B)
  ri = return_value(*,ai,bi)
  array = Vector{typeof(ri)}(undef,param_length(a))
  GenericParamUnit(array,_prod_innersize(a,b))
end

function LinearAlgebra.rmul!(a::GenericParamUnit,β)
  for i in eachindex(a.array)
    rmul!(a.array[i],β)
  end
end

function Fields._zero_entries!(a::GenericParamUnit)
  for i in eachindex(a.array)
    Fields._zero_entries!(a.array[i])
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

  for i in eachindex(c.array)
    mul!(param_getindex(c,i),param_getindex(a,i),param_getindex(b,i),α,β)
  end
end

function Fields._setsize_mul!(c,a::ParamUnit,b::ParamUnit)
  for i in eachindex(c.array)
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
  for i in eachindex(a.array)
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

function Arrays.return_cache(k::Arrays.ConfigMap{typeof(ForwardDiff.gradient)},x::GenericParamUnit{A,1}) where A
  xi = testitem(x)
  fi = return_cache(k,xi)
  array = Vector{typeof(fi)}(undef,length(x.array))
  for i in eachindex(x.array)
    array[i] = return_cache(k,x.array[i])
  end
  GenericParamUnit(array,x.innersize)
end

function Arrays.return_cache(k::Arrays.ConfigMap{typeof(ForwardDiff.jacobian)},x::GenericParamUnit)
  xi = testitem(x)
  fi = return_cache(k,xi)
  array = Vector{typeof(fi)}(undef,length(x.array))
  for i in eachindex(x.array)
    array[i] = return_cache(k,x.array[i])
  end
  GenericParamUnit(array,x.innersize)
end

function Arrays.return_cache(k::Arrays.DualizeMap,x::GenericParamUnit)
  cfg = return_cache(Arrays.ConfigMap(k.f),x)
  xi = testitem(x)
  cfgi = testitem(cfg)
  xidual = evaluate!(cfgi,k,xi)
  array = Vector{typeof(xidual)}(undef,length(x.array))
  cfg,GenericParamUnit(array,x.innersize)
end

function Arrays.evaluate!(cache,k::Arrays.DualizeMap,x::GenericParamUnit)
  cfg, xdual = cache
  for i in eachindex(x.array)
    xdual.array[i] = evaluate!(cfg.array[i],k,x.array[i])
  end
  xdual
end

function Arrays.return_cache(k::Arrays.AutoDiffMap,ydual::GenericParamUnit,x,cfg::GenericParamUnit)
  yidual = testitem(ydual)
  xi = testitem(x)
  cfgi = testitem(cfg)
  ci = return_cache(k,yidual,xi,cfgi)
  ri = evaluate!(ci,k,yidual,xi,cfgi)
  c = Vector{typeof(ci)}(undef,length(ydual.array))
  array = Vector{typeof(ri)}(undef,length(ydual.array))
  for i in eachindex(ydual.array)
    c[i] = return_cache(k,ydual.array[i],x.array[i],cfg.array[i])
  end
  GenericParamUnit(array,ydual.innersize),c
end

function Arrays.evaluate!(cache,k::Arrays.AutoDiffMap,ydual::GenericParamUnit,x,cfg::GenericParamUnit)
  r,c = cache
  for i in eachindex(ydual.array)
    r.array[i] = evaluate!(c[i],k,ydual.array[i],x.array[i],cfg.array[i])
  end
  r
end

# block map interface

function Arrays.return_value(k::Map,f::ArrayBlock{A,N},h::ParamUnit) where {A,N}
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

function Arrays.return_cache(k::Map,f::ArrayBlock{A,N},h::ParamUnit) where {A,N}
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

function Arrays.evaluate(cache,k::Map,f::ArrayBlock,h::ParamUnit)
  g,l = cache
  @check g.touched == f.touched
  for i in eachindex(f.array)
    if f.touched[i]
      g.array[i] = evaluate!(l[i],k,f.array[i],h)
    end
  end
  g
end

function Arrays.return_value(k::Map,h::ParamUnit,f::ArrayBlock{A,N}) where {A,N}
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

function Arrays.return_cache(k::Map,h::ParamUnit,f::ArrayBlock{A,N}) where {A,N}
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

function Arrays.evaluate(cache,k::Map,h::ParamUnit,f::ArrayBlock)
  g,l = cache
  @check g.touched == f.touched
  for i in eachindex(f.array)
    if f.touched[i]
      g.array[i] = evaluate!(l[i],k,h,f.array[i])
    end
  end
  g
end

# constructors

function Fields.GenericField(f::AbstractParamFunction)
  GenericParamUnit(map(i -> GenericField(f[i]),1:length(f)))
end
