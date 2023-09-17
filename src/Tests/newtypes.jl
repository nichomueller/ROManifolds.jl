struct SupportQuantity{T}
  array::AbstractVector{<:AbstractArray{T}}

  function SupportQuantity(array::AbstractVector{<:AbstractArray{T}}) where T
    new{T}(array)
  end

  function SupportQuantity(a::S,length::Int) where S
    SupportQuantity(fill(a,length))
  end
end

Base.size(a::SupportQuantity,i...) = size(testitem(a),i...)
Base.length(a::SupportQuantity) = length(a.array)
Base.eltype(::Type{SupportQuantity{T}}) where T = T
Base.eltype(::SupportQuantity{T}) where T = T
Base.eachindex(a::SupportQuantity) = eachindex(a.array)

function Base.getindex(a::SupportQuantity,i...)
  a.array[i...]
end

function Base.setindex!(a::SupportQuantity,v,i...)
  a.array[i...] = v
end

function Base.first(a::SupportQuantity)
  SupportQuantity([first(testitem(a))])
end

function Arrays.CachedArray(a::SupportQuantity)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  for i in eachindex(a.array)
    array[i] = CachedArray(a.array[i])
  end
  SupportQuantity(array)
end

Arrays.testitem(a::SupportQuantity) = testitem(a.array)

function Arrays.return_value(f,a::SupportQuantity,x::Union{AbstractArray,SupportQuantity}...)
  a1 = get_at_index(1,a,x...)
  return_value(f,a1)
end

function Arrays.return_cache(f,a::SupportQuantity,x::Union{AbstractArray,SupportQuantity}...)
  sq1 = _get_1st_sq(a,x...)
  n = length(sq1)
  val = return_value(f,a,x...)
  sval = SupportQuantity(val,n)
  a1 = get_at_index(1,a,x...)
  cx = return_cache(f,a1)
  cx,sval
end

function Arrays.evaluate!(cache,f,a::SupportQuantity,x::Union{AbstractArray,SupportQuantity}...)
  cx,sval = cache
  @inbounds for i = eachindex(sval)
    ai = get_at_index(i,a,x...)
    sval.array[i] = evaluate!(cx,f,ai...)
  end
  sval
end

function Arrays.return_value(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::SupportQuantity,
  x::Vararg{Union{AbstractArray,SupportQuantity}})

  a1 = get_at_index(1,a,x...)
  return_value(f,a1...)
end

function Arrays.return_cache(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::SupportQuantity,
  x::Vararg{Union{AbstractArray,SupportQuantity}})

  sq1 = _get_1st_sq(a,x...)
  n = length(sq1)
  val = return_value(f,a,x...)
  sval = SupportQuantity(val,n)
  a1 = get_at_index(1,a,x...)
  cx = return_cache(f,a1...)
  cx,sval
end

function Arrays.evaluate!(
  cache,
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::SupportQuantity,
  x::Vararg{Union{AbstractArray,SupportQuantity}})

  cx,sval = cache
  @inbounds for i = eachindex(sval)
    ai = get_at_index(i,a,x...)
    sval.array[i] = evaluate!(cx,f,ai...)
  end
  sval
end

function Arrays.return_value(f,a::AbstractArray,x::SupportQuantity)
  a1 = get_at_index(1,a,x)
  return_value(f,a1)
end

function Arrays.return_cache(f,a::AbstractArray,x::SupportQuantity)
  n = length(x)
  val = return_value(f,a,x)
  sval = SupportQuantity(val,n)
  a1 = get_at_index(1,a,x)
  cx = return_cache(f,a1...)
  cx,sval
end

function Arrays.evaluate!(cache,f,a::AbstractArray,x::SupportQuantity)
  cx,sval = cache
  @inbounds for i = eachindex(sval)
    ai = get_at_index(i,a,x)
    sval.array[i] = evaluate!(cx,f,ai...)
  end
  sval
end

function get_at_index(i::Int,x::Union{AbstractArray,SupportQuantity}...)
  ret = ()
  @inbounds for xj in x
    ret = isa(xj,SupportQuantity) ? (ret...,xj[i]) : (ret...,xj)
  end
  return ret
end

function _get_1st_sq(x::Union{AbstractArray,SupportQuantity}...)
  for xi in x
    if isa(xi,SupportQuantity)
      return xi
    end
  end
end

function Arrays.lazy_map(k,a::SupportQuantity,x::Union{AbstractArray,SupportQuantity}...)
  lazy_arrays = map(eachindex(a)) do i
    ai = get_at_index(i,a,x...)
    lazy_map(k,ai...)
  end
  SupportQuantity(lazy_arrays)
end

function Arrays.lazy_map(k,a::SupportQuantity,b::Vararg{Union{AbstractArray,SupportQuantity}})
  lazy_arrays = map(eachindex(a)) do i
    ai = get_at_index(i,a,b...)
    lazy_map(k,ai...)
  end
  SupportQuantity(lazy_arrays)
end

function Arrays.lazy_map(k,a::AbstractArray,x::SupportQuantity)
  lazy_arrays = map(y->lazy_map(k,a,y),x.array)
  SupportQuantity(lazy_arrays)
end

struct PTField{T} <: Field
  object::T
end

Arrays.testargs(a::PTField,x::Point) = testargs(a.object,x)
Arrays.return_value(a::PTField,x::Point) = return_value(a.object,x)
Arrays.return_cache(a::PTField,x::Point) = return_cache(a.object,x)
Arrays.evaluate!(cache,a::PTField,x::Point) = evaluate!(cache,a.object,x)

function get_pt_fields(ptf::PTFunction)
  p,t = ptf.params,ptf.times
  np = length(p)
  nt = length(t)
  npt = np*nt
  fields = Vector{PTField}(undef,npt)
  @inbounds for k = 1:npt
    pk = p[fast_idx(k,np)]
    tk = t[slow_idx(k,np)]
    fields[k] = PTField(ptf.f(pk,tk))
  end
  fields
end

struct NewGenericPTCellField{DS} <: PTCellField
  cell_field::SupportQuantity
  trian::Triangulation
  domain_style::DS
  function NewGenericPTCellField(
    cell_field::SupportQuantity,
    trian::Triangulation,
    domain_style::DomainStyle)

    DS = typeof(domain_style)
    new{DS}(Fields.MemoArray(cell_field),trian,domain_style)
  end
end

Base.length(f::NewGenericPTCellField) = length(f.cell_field)
CellData.get_data(f::NewGenericPTCellField) = f.cell_field
CellData.get_triangulation(f::NewGenericPTCellField) = f.trian
CellData.DomainStyle(::Type{NewGenericPTCellField{DS}}) where DS = DS()
function CellData.similar_cell_field(::PTCellField,cell_data,trian,ds)
  NewGenericPTCellField(cell_data,trian,ds)
end

function CellData.CellField(f::PTFunction,trian::Triangulation,::DomainStyle)
  s = size(get_cell_map(trian))
  cell_field = SupportQuantity(lazy_map(x->Fill(x,s),get_pt_fields(f)))
  NewGenericPTCellField(cell_field,trian,PhysicalDomain())
end

op = feop
μ = realization(op,2)
t = dt
x = get_cell_points(dΩ.quad)
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))

q = aμt(μ,t)*∇(dv)
resq = q(x)

q_ok = a(μ[1],t)*∇(dv)
resq_ok = q_ok(x)

function test_ptarray(a::SupportQuantity,b::AbstractArray)
  a1 = testitem(a)
  @assert typeof(a1) == typeof(b)
  @assert all(a1 .≈ b)
  return
end

function test_ptarray(a::AbstractArray,b::SupportQuantity)
  test_ptarray(b,a)
end

test_ptarray(resq,resq_ok)
