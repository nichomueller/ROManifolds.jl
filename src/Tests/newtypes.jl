struct SupportQuantity{T}
  array::Vector{T}

  function SupportQuantity(array::Array{T}) where {T}
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

function Arrays.CachedArray(a::SupportQuantity)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  for i in eachindex(a.array)
    array[i] = CachedArray(a.array[i])
  end
  ArrayBlock(array)
end

Arrays.testitem(a::SupportQuantity) = testitem(a.array)

function Arrays.lazy_map(k,a::SupportQuantity)
  lazy_arrays = map(x->lazy_map(k,x),a.array)
  SupportQuantity(lazy_arrays)
end

function Arrays.lazy_map(k,a::SupportQuantity,b::AbstractArray...)
  lazy_arrays = map(x->lazy_map(k,x,b...),a.array)
  SupportQuantity(lazy_arrays)
end

function Arrays.lazy_map(k,a::AbstractArray,b::SupportQuantity)
  lazy_arrays = map(x->lazy_map(k,a,x),b.array)
  SupportQuantity(lazy_arrays)
end

function Arrays.lazy_map(k,a::SupportQuantity,b::SupportQuantity)
  lazy_arrays = map((x,y)->lazy_map(k,x,y),a.array,b.array)
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
  cell_field::AbstractArray
  trian::Triangulation
  domain_style::DS
  function NewGenericPTCellField(
    cell_field::AbstractArray,
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
  cell_field = lazy_map(x->Fill(x,s),get_pt_fields(f))
  NewGenericPTCellField(cell_field,trian,PhysicalDomain())
end

# cf = CellField(aμt(μ,t),Ω,ReferenceDomain())
# cfx = cf(x)

op = feop
μ = realization(op,2)
t = dt
x = get_cell_points(dΩ.quad)
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))

q = aμt(μ,t)*∇(dv)
resq = q(x)

# # ax = map(i->i(x),q.args[1])
# _f,_x = CellData._to_common_domain(q.args[1],x)
# cell_field = get_data(_f)
# cell_point = get_data(_x)
# # lazy_map(evaluate,cell_field,cell_point)
# ff = cell_field.args[1]
# gg = cell_field.args[2]
# gx = lazy_map(evaluate,gg,cell_point)
# # fx = lazy_map(evaluate,ff,gx)
# fi = map(testitem,(ff,gx))
# # T = return_type(evaluate,fi...) # @which evaluate!(nothing,fi...) -> Matrix
# cache = return_cache(evaluate,fi[1],gx[1])
# evaluate!(cache,evaluate,fi[1],gx[1])

# q_ok = a(μ[1],t)*∇(dv)
# resq_ok = q_ok(x)
# # ax_ok = map(i->i(x),q_ok.args[1])
# _f_ok,_x_ok = CellData._to_common_domain(q_ok.args[1],x)
# cell_field_ok = get_data(_f_ok)
# cell_point_ok = get_data(_x_ok)
# # lazy_map(evaluate,cell_field_ok,cell_point_ok)
# ff_ok = cell_field_ok.args[1]
# gg_ok = cell_field_ok.args[2]
# gx_ok = lazy_map(evaluate,gg_ok,cell_point_ok)
# # fx_ok = lazy_map(evaluate,ff_ok,gx_ok)
# fi_ok = map(testitem,(ff_ok,gx_ok))
# T_ok = return_type(evaluate,fi_ok...) # @which evaluate!(nothing,fi_ok...) -> Vector
# cache_ok = return_cache(evaluate,fi_ok[1],gx_ok[1])
# evaluate!(cache_ok,evaluate,fi_ok[1],gx_ok[1])

# typeof(resq[1]) == typeof(resq_ok)
# all(resq[1] .≈ resq_ok)

# q = aμt(μ,t)*∇(dv)⋅∇(du)
# resq = q(x)
# resq_ok = (a(μ[1],t)*∇(dv)⋅∇(du))(x)
# typeof(resq[1]) == typeof(resq_ok)
# all(resq[1] .≈ resq_ok)
