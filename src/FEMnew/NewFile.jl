abstract type AbstractPTFunction{P,T} <: Function end

struct PFunction{P} <: AbstractPTFunction{P,Nothing}
  f::Function
  params::P
end

struct PTFunction{P,T} <: AbstractPTFunction{P,T}
  f::Function
  params::P
  times::T
end

function get_fields(pf::PFunction{<:AbstractVector{<:Number}})
  p = pf.params
  GenericField(pf.f(p))
end

function get_fields(pf::PFunction)
  p = pf.params
  np = length(p)
  fields = Vector{GenericField}(undef,np)
  @inbounds for k = eachindex(p)
    pk = p[k]
    fields[k] = GenericField(pf.f(pk))
  end
  fields
end

function get_fields(ptf::PTFunction{<:AbstractVector{<:Number},<:Real})
  p,t = ptf.params,ptf.times
  GenericField(ptf.f(p,t))
end

function get_fields(ptf::PTFunction{<:AbstractVector{<:Number},<:AbstractVector{<:Number}})
  p,t = ptf.params,ptf.times
  nt = length(t)
  fields = Vector{GenericField}(undef,nt)
  @inbounds for k = 1:nt
    tk = t[k]
    fields[k] = GenericField(ptf.f(p,tk))
  end
  fields
end

function get_fields(ptf::PTFunction)
  p,t = ptf.params,ptf.times
  np = length(p)
  nt = length(t)
  npt = np*nt
  fields = Vector{GenericField}(undef,npt)
  @inbounds for k = 1:npt
    pk = p[slow_idx(k,nt)]
    tk = t[fast_idx(k,nt)]
    fields[k] = GenericField(ptf.f(pk,tk))
  end
  fields
end

function Arrays.evaluate!(cache,f::AbstractPTFunction,x::Point)
  g = get_fields(f)
  map(g) do gi
    gi(x)
  end
end

const AbstractArrayBlock{T,N} = Union{AbstractArray{T,N},ArrayBlock{T,N}}

struct Nonaffine <: OperatorType end

struct PTArray{T,N}
  array::AbstractArrayBlock{T,N}
end

Arrays.get_array(a::PTArray) = a.array
Base.size(a::PTArray,i...) = size(testitem(a),i...)
Base.eltype(::Type{PTArray{T}}) where T = eltype(T)
Base.eltype(::PTArray{T}) where T = eltype(T)
Base.ndims(::PTArray{T,N} where T) where N = N
Base.ndims(::Type{<:PTArray{T,N}} where T) where N = N
Base.first(a::PTArray) = first(testitem(a))
Base.length(a::PTArray) = length(a.array)
Base.eachindex(a::PTArray) = eachindex(a.array)
Base.lastindex(a::PTArray) = lastindex(a.array)
Base.getindex(a::PTArray,i...) = a.array[i...]
Base.setindex!(a::PTArray,v,i...) = a.array[i...] = v
Base.iterate(a::PTArray,i...) = iterate(a.array,i...)

function Base.show(io::IO,::MIME"text/plain",a::PTArray{T}) where T
  println(io, "PTArray with eltype $T and elements")
  for i in eachindex(a)
    println(io,"  ",a.array[i])
  end
end

function Base.copy(a::PTArray{T}) where T
  b = Vector{T}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = copy(a[i])
  end
  PTArray(b)
end

function Base.similar(a::PTArray{T}) where T
  b = Vector{T}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = similar(a[i])
  end
  PTArray(b)
end

function Base.zero(a::PTArray)
  T = eltype(a)
  b = similar(a)
  b .= zero(T)
end

function Base.zeros(a::PTArray)
  get_array(zero(a))
end

function Base.sum(a::PTArray)
  sum(a.array)
end

for op in (:+,:-,:*)
  @eval begin
    function ($op)(a::PTArray,b::PTArray)
      map($op,a,b)
    end
  end
end

function Base.:*(a::PTArray,b::Number)
  map(ai->(*)(ai,b),a)
end

function Base.:*(a::Number,b::PTArray)
  b*a
end

function Base.:≈(a::PTArray,b::PTArray)
  @assert size(a) == size(b)
  for i in eachindex(a)
    if !(a[i] ≈ b[i])
      return false
    end
  end
  true
end

function Base.:(==)(a::PTArray,b::PTArray)
  @assert size(a) == size(b)
  for i in eachindex(a)
    if !(a[i] == b[i])
      return false
    end
  end
  true
end

function Base.transpose(a::PTArray)
  map(transpose,a)
end

function Base.hcat(a::PTArray...)
  n = length(first(a))
  harray = map(1:n) do j
    arrays = ()
    @inbounds for i = eachindex(a)
      arrays = (arrays...,a[i][j])
    end
    hcat(arrays...)
  end
  PTArray(harray)
end

function Base.vcat(a::PTArray...)
  n = length(first(a))
  varray = map(1:n) do j
    arrays = ()
    @inbounds for i = eachindex(a)
      arrays = (arrays...,a[i][j])
    end
    vcat(arrays...)
  end
  PTArray(varray)
end

function Base.stack(a::PTArray...)
  n = length(first(a))
  harray = map(1:n) do j
    arrays = ()
    @inbounds for i = eachindex(a)
      arrays = (arrays...,a[i][j])
    end
    stack(arrays)
  end
  PTArray(harray)
end

function Base.hvcat(nblocks::Int,a::PTArray...)
  nrows = Int(length(a)/nblocks)
  varray = map(1:nrows) do row
    vcat(a[(row-1)*nblocks+1:row*nblocks]...)
  end
  hvarray = hcat(varray...)
  hvarray
end

function Base.fill!(a::PTArray,z)
  @inbounds for i = eachindex(a)
    ai = a[i]
    fill!(ai,z)
  end
end

function LinearAlgebra.fillstored!(a::PTArray,z)
  @inbounds for i = eachindex(a)
    ai = a[i]
    fillstored!(ai,z)
  end
end

function Arrays.CachedArray(a::PTArray)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  @inbounds for i in eachindex(a)
    array[i] = CachedArray(a.array[i])
  end
  PTArray(array)
end

function Base.map(f,a::PTArray)
  n = length(a)
  fa1 = f(testitem(a))
  array = Vector{typeof(fa1)}(undef,n)
  @inbounds for i = 1:n
    array[i] = f(a[i])
  end
  PTArray(array)
end

function Base.map(f,a::PTArray,x::Union{AbstractArrayBlock,PTArray}...)
  n = _get_length(a,x...)
  ax1 = get_at_index(1,(a,x...))
  fax1 = f(ax1...)
  array = Vector{typeof(fax1)}(undef,n)
  @inbounds for i = 1:n
    axi = get_at_index(i,(a,x...))
    array[i] = f(axi...)
  end
  PTArray(array)
end

function Base.map(f,a::AbstractArrayBlock,x::PTArray)
  n = length(x)
  fax1 = f(a,testitem(x))
  array = Vector{typeof(fax1)}(undef,n)
  @inbounds for i = 1:n
    array[i] = f(a,x[i])
  end
  PTArray(array)
end

struct PTBroadcasted{T}
  array::PTArray{T}
end
_get_pta(a::PTArray) = a
_get_pta(a::PTBroadcasted) = a.array

function Base.broadcasted(f,a::Union{PTArray,PTBroadcasted}...)
  pta = map(_get_pta,a)
  PTBroadcasted(map(f,pta...))
end

function Base.broadcasted(f,a::Number,b::Union{PTArray,PTBroadcasted})
  PTBroadcasted(map(x->f(a,x),_get_pta(b)))
end

function Base.broadcasted(f,a::Union{PTArray,PTBroadcasted},b::Number)
  PTBroadcasted(map(x->f(x,b),_get_pta(a)))
end

function Base.broadcasted(
  f,a::Union{PTArray,PTBroadcasted},
  b::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,a::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}},
  b::Union{PTArray,PTBroadcasted})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::PTBroadcasted{T}) where T
  a = similar(b)
  Base.materialize!(a,b)
  a
end

function Base.materialize!(a::PTArray,b::Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),a)
  a
end

function Base.materialize!(a::PTArray,b::PTBroadcasted)
  map(Base.materialize!,a,b.array)
  a
end

function Arrays.testitem(a::PTArray{T}) where T
  if length(a) != 0
    a[1]
  else
    fill(eltype(a),1)
  end
end

function Arrays.setsize!(a::PTArray{<:CachedArray},size::Tuple{Vararg{Int}})
  @inbounds for i in eachindex(a)
    setsize!(a[i],size)
  end
end

function LinearAlgebra.ldiv!(a::PTArray,m::LU,b::PTArray)
  @check length(a) == length(b)
  @inbounds for i = eachindex(a)
    ai,bi = a[i],b[i]
    ldiv!(ai,m,bi)
  end
end

function Arrays.get_array(a::PTArray{<:CachedArray})
  map(x->getproperty(x,:array),a)
end

function get_at_offsets(x::PTArray{<:AbstractVector},offsets::Vector{Int},row::Int)
  map(y->y[offsets[row]+1:offsets[row+1]],x)
end

function get_at_offsets(x::PTArray{<:AbstractMatrix},offsets::Vector{Int},row::Int,col::Int)
  map(y->y[offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1]],x)
end

get_at_index(::Int,x) = x

get_at_index(i::Int,x::PTArray) = x[i]

function get_at_index(i::Int,x::NTuple{N,Union{AbstractArrayBlock,PTArray}}) where N
  ret = ()
  @inbounds for xj in x
    ret = (ret...,get_at_index(i,xj))
  end
  return ret
end

function get_at_index(range::UnitRange{Int},x::NTuple{N,PTArray}) where N
  ret = ()
  @inbounds for j in range
    ret = (ret...,get_at_index(j,x))
  end
  return ret
end

function get_at_index(::Colon,x::NTuple{N,PTArray}) where N
  ret = ()
  @inbounds for j in eachindex(first(x))
    ret = (ret...,get_at_index(j,x))
  end
  return ret
end

function _get_length(x::Union{AbstractArrayBlock,PTArray}...)
  pta = filter(y->isa(y,PTArray),x)
  n = length(first(pta))
  @check all([length(y) == n for y in pta])
  n
end

struct PTGenericField
  fields::AbstractVector{GenericField}
  function PTGenericField(f::PTFunction)
    fields = get_fields(f)
    new(fields)
  end
  function PTGenericField(f::AbstractVector{<:Function})
    fields = GenericField.(f)
    new(fields)
  end
end

Base.size(a::PTGenericField) = size(a.fields)
Base.length(a::PTGenericField) = length(a.fields)
Base.IndexStyle(::Type{<:PTGenericField}) = IndexLinear()
Base.getindex(a::PTGenericField,i::Integer) = GenericField(a.fields[i])
Base.iterate(a::PTGenericField,i...) = iterate(a.fields,i...)

function Arrays.testitem(f::PTGenericField)
  f[1]
end

function Arrays.return_cache(f::PTGenericField,x)
  fi = testitem(f)
  li = return_cache(fi,x)
  fix = evaluate!(li,fi,x)
  l = Vector{typeof(li)}(undef,size(f.fields))
  g = Vector{typeof(fix)}(undef,size(f.fields))
  for i in eachindex(f.fields)
    l[i] = return_cache(f.fields[i],x)
  end
  PTArray(g),l
end

function Arrays.evaluate!(cache,f::PTGenericField,x)
  g,l = cache
  for i in eachindex(f.fields)
    g.array[i] = evaluate!(l[i],f.fields[i],x)
  end
  g
end

abstract type PTCellField <: CellField end

struct GenericPTCellField{DS} <: PTCellField
  cell_field::AbstractArray
  trian::Triangulation
  domain_style::DS
  function GenericPTCellField(
    cell_field::AbstractArray,
    trian::Triangulation,
    domain_style::DomainStyle)

    DS = typeof(domain_style)
    new{DS}(Fields.MemoArray(cell_field),trian,domain_style)
  end
end

CellData.get_data(f::GenericPTCellField) = f.cell_field
FESpaces.get_triangulation(f::GenericPTCellField) = f.trian
CellData.DomainStyle(::Type{GenericPTCellField{DS}}) where DS = DS()

function CellData.similar_cell_field(::PTCellField,cell_data,trian,ds)
  GenericPTCellField(cell_data,trian,ds)
end

function CellData.CellField(f::PTFunction,trian::Triangulation,::DomainStyle)
  s = size(get_cell_map(trian))
  cell_field = Fill(PTGenericField(f),s)
  GenericPTCellField(cell_field,trian,PhysicalDomain())
end

struct PTOperationCellField{DS} <: PTCellField
  op::Operation
  args::Tuple
  trian::Triangulation
  domain_style::DS
  memo::Dict{Any,Any}

  function PTOperationCellField(op::Operation,args::CellField...;initial_check=false)
    @assert length(args) > 0
    trian = get_triangulation(first(args))
    domain_style = DomainStyle(first(args))
    @check all( map(i->DomainStyle(i)==domain_style,args) )
    if initial_check && num_cells(trian)>0
      x = _get_cell_points(args...)
      ax = map(i->i(x),args)
      axi = map(first,ax)
      Fields.BroadcastingFieldOpMap(op.op)(axi...)
    end
    new{typeof(domain_style)}(op,args,trian,domain_style,Dict())
  end
end

function CellData._get_cell_points(args::CellField...)
  k = findfirst(i->isa(i,CellState),args)
  if isnothing(k)
    j = findall(i->isa(i,Union{OperationCellField,PTOperationCellField}),args)
    if length(j) == 0
      CellData._get_cell_points(first(args))
    else
      CellData._get_cell_points(args[j]...)
    end
  else
    args[k].points
  end
end

function CellData._get_cell_points(a::PTOperationCellField...)
  b = []
  for ai in a
    for i in ai.args
      push!(b,i)
    end
  end
  CellData._get_cell_points(b...)
end

function CellData._get_cell_points(a::PTOperationCellField)
  CellData._get_cell_points(a.args...)
end

function CellData.get_data(f::PTOperationCellField)
  a = map(get_data,f.args)
  lazy_map(Broadcasting(f.op),a...)
end

FESpaces.get_triangulation(f::PTOperationCellField) = f.trian
CellData.DomainStyle(::Type{PTOperationCellField{DS}}) where DS = DS()

function Arrays.evaluate!(cache,f::PTOperationCellField,x::CellPoint)
  ax = map(i->i(x),f.args)
  lazy_map(BroadcastingFieldOpMap(f.op.op),ax...)
end

function CellData.change_domain(
  f::PTOperationCellField,
  target_trian::Triangulation,
  target_domain::DomainStyle)

  args = map(i->change_domain(i,target_trian,target_domain),f.args)
  PTOperationCellField(f.op,args...)
end

function CellData._operate_cellfields(k::Operation,a...)
  b = _to_common_domain(a...)
  if any(x->isa(x,PTCellField),b)
    PTOperationCellField(k,b...)
  else
    OperationCellField(k,b...)
  end
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::PTArray,
  x::Vararg{AbstractArray})

  n = _get_length(a,x...)
  ax1 = get_at_index(1,(a,x...))
  v1 = return_value(f,ax1...)
  array = Vector{typeof(v1)}(undef,n)
  for i = 1:n
    axi = get_at_index(i,(a,x...))
    array[i] = return_value(f,axi...)
  end
  PTArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::PTArray,
  x::Vararg{AbstractArray})

  n = _get_length(a,x...)
  ax1 = get_at_index(1,(a,x...))
  c1 = return_cache(f,ax1...)
  b1 = evaluate!(c1,f,ax1...)
  cache = Vector{typeof(c1)}(undef,n)
  array = Vector{typeof(b1)}(undef,n)
  for i = 1:n
    axi = get_at_index(i,(a,x...))
    cache[i] = return_cache(f,axi...)
  end
  cache,array
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray{<:AbstractArray{T,N}},
  b::AbstractArray{S,N}) where {T,S,N}

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray{<:AbstractMatrix},
  b::AbstractArray{S,3} where S)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  b::PTArray{<:AbstractArray{S,3} where S},
  a::AbstractMatrix)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,b[i],a)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray{<:AbstractVector},
  b::AbstractMatrix)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  b::PTArray{<:AbstractMatrix},
  a::AbstractVector)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,b[i],a)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray{<:AbstractVector},
  b::AbstractArray{S,3} where S)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  b::PTArray{<:AbstractArray{S,3} where S},
  a::AbstractVector)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,b[i],a)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray,
  x::Vararg{AbstractArray})

  cx,array = cache
  @inbounds for i = eachindex(array)
    axi = get_at_index(i,(a,x...))
    array[i] = evaluate!(cx[i],f,axi...)
  end
  PTArray(array)
end

function Arrays.return_value(
  f::IntegrationMap,
  a::PTArray,
  x::Vararg{AbstractArray})

  n = _get_length(a,x...)
  ax1 = get_at_index(1,(a,x...))
  v1 = return_value(f,ax1...)
  array = Vector{typeof(v1)}(undef,n)
  for i = 1:n
    axi = get_at_index(i,(a,x...))
    array[i] = return_value(f,axi...)
  end
  PTArray(array)
end

function Arrays.return_cache(
  f::IntegrationMap,
  a::PTArray,
  x::Vararg{AbstractArray})

  n = _get_length(a,x...)
  ax1 = get_at_index(1,(a,x...))
  c1 = return_cache(f,ax1...)
  b1 = evaluate!(c1,f,ax1...)
  cache = Vector{typeof(c1)}(undef,n)
  array = Vector{typeof(b1)}(undef,n)
  for i = 1:n
    axi = get_at_index(i,(a,x...))
    cache[i] = return_cache(f,axi...)
  end
  cache,array
end

function Arrays.evaluate!(
  cache,
  f::IntegrationMap,
  a::PTArray,
  x::Vararg{AbstractArray})

  cx,array = cache
  @inbounds for i = eachindex(array)
    axi = get_at_index(i,(a,x...))
    array[i] = evaluate!(cx[i],f,axi...)
  end
  PTArray(array)
end
