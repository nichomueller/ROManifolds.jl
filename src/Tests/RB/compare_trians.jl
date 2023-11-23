struct MyReindex{A} <: Map
  values::A
end

function Arrays.return_value(k::MyReindex,i...)
  length(k.values)!=0 ? evaluate(k,testargs(k,i...)...) : testitem(k.values)
end
Arrays.return_cache(k::MyReindex,i...) = array_cache(k.values)
Arrays.evaluate!(cache,k::MyReindex,i...) = getindex!(cache,k.values,i...)
Arrays.evaluate(k::MyReindex,i...) = k.values[i...]

Arrays.testitem(k::MyReindex{<:PTArray}) = MyReindex(testitem(k.values))
function Arrays.return_value(k::MyReindex{<:PTArray},i...)
  return_value(testitem(k),i...)
end
function Arrays.return_cache(k::MyReindex{<:PTArray},i...)
  cache = return_cache(testitem(k))
  val = return_value(k,i...)
  array = Vector{typeof(val)}(undef,length(k.values))
  return cache,array
end
function Arrays.evaluate!(cache,k::MyReindex{<:NonaffinePTArray},i...)
  cache,array = cache
  for n in eachindex(array)
    kn = MyReindex(k.values[n])
    array[n] = evaluate!(cache,kn,i...)
  end
  NonaffinePTArray(array)
end
function Arrays.evaluate!(cache,k::MyReindex{<:AffinePTArray},i...)
  cache,array = cache
  evaluate!(cache,testitem(k),i...)
  AffinePTArray(cache.array,length(array))
end
function Arrays.evaluate(k::MyReindex{<:PTArray},i...)
  cache = return_cache(k,i...)
  evaluate!(cache,k,i...)
end

function Arrays.lazy_map(k::MyReindex{<:Table},::Type{T},j_to_i::AbstractArray) where T
  i_to_v = k.values
  Table(i_to_v[j_to_i])
end

function Arrays.lazy_map(k::MyReindex{<:CompressedArray},::Type{T},j_to_i::AbstractArray) where T
  i_to_v = k.values
  values = i_to_v.values
  ptrs = i_to_v.ptrs[j_to_i]
  CompressedArray(values,ptrs)
end

function Arrays.lazy_map(k::MyReindex{<:LazyArray},::Type{T},j_to_i::AbstractArray) where T
  i_to_maps = k.values.maps
  i_to_args = k.values.args
  j_to_maps = lazy_map(MyReindex(i_to_maps),eltype(i_to_maps),j_to_i)
  j_to_args = map(i_to_fk->lazy_map(MyReindex(i_to_fk),eltype(i_to_fk),j_to_i), i_to_args)
  LazyArray(T,j_to_maps,j_to_args...)
end

function Arrays.lazy_map(k::MyReindex{<:PTArray},j_to_i::AbstractArray)
  map(value -> lazy_map(MyReindex(value),j_to_i),k.values)
end

abstract type ChildGrid{Dc,Dp} <: Grid{Dc,Dp} end

struct ChildUnstructuredGrid{Dc,Dp,Tp,O,Tn} <: ChildGrid{Dc,Dp}
  parent_id::UInt
  node_coordinates::Vector{Point{Dp,Tp}}
  cell_node_ids::Table{Int32,Vector{Int32},Vector{Int32}}
  reffes::Vector{LagrangianRefFE{Dc}}
  cell_types::Vector{Int8}
  orientation_style::O
  facet_normal::Tn
  cell_map
  function ChildUnstructuredGrid(
    parent::UnstructuredGrid{Dc,Dp,Tp,O,Tn},
    cell_to_parent_cell::AbstractArray) where {Dc,Dp,Tp,O,Tn}

    parent_id = objectid(parent)
    node_coordinates = get_node_coordinates(parent)
    cell_node_ids = Table(get_cell_node_ids(parent)[cell_to_parent_cell])
    reffes = get_reffes(parent)
    cell_types = get_cell_type(parent)[cell_to_parent_cell]
    orientation_style = parent.orientation_style
    facet_normal = nothing
    cell_map = lazy_map(MyReindex(get_cell_map(parent)),cell_to_parent_cell)
    new{Dc,Dp,Tp,O,Tn}(parent_id,node_coordinates,cell_node_ids,reffes,cell_types,
    orientation_style,facet_normal,cell_map)
  end
end

function _is_parent(strian::BodyFittedTriangulation,ttrian::BodyFittedTriangulation)
  objectid(get_grid(strian)) == get_grid(ttrian).parent_id
end

myview(a::Grid,b::AbstractArray) = ChildUnstructuredGrid(a,b)

function myview(t::BodyFittedTriangulation,ids::AbstractArray)
  model = t.model
  grid = myview(t.grid,ids)
  tface_to_mface = t.tface_to_mface[ids]
  BodyFittedTriangulation(model,grid,tface_to_mface)
end

function Geometry.OrientationStyle(::Type{ChildUnstructuredGrid{Dc,Dp,G}}) where {Dc,Dp,G}
  OrientationStyle(G)
end

function Geometry.RegularityStyle(::Type{ChildUnstructuredGrid{Dc,Dp,G}}) where {Dc,Dp,G}
  RegularityStyle(G)
end

function Geometry.get_node_coordinates(grid::ChildUnstructuredGrid)
  grid.node_coordinates
end

function Geometry.get_cell_node_ids(grid::ChildUnstructuredGrid)
  grid.cell_node_ids
end

function Geometry.get_reffes(grid::ChildUnstructuredGrid)
  grid.reffes
end

function Geometry.get_cell_type(grid::ChildUnstructuredGrid)
  grid.cell_types
end

function Geometry.get_cell_map(grid::ChildUnstructuredGrid)
  grid.cell_map
end

function Geometry.get_facet_normal(grid::ChildUnstructuredGrid)
  grid.facet_normal
end

function CellData.change_domain(
  a::CellField,
  strian::BodyFittedTriangulation,::ReferenceDomain,
  ttrian::BodyFittedTriangulation{D,Dp,A,<:ChildUnstructuredGrid,C},::ReferenceDomain
  ) where {D,Dp,A,C}
  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  if strian === ttrian
    return a
  elseif _is_parent(strian,ttrian)
    sface_to_field = get_data(a)
    sglue = get_glue(strian,Val(D))
    mface_to_sface = sglue.mface_to_tface
    tface_to_mface = ttrian.tface_to_mface
    mface_to_field = extend(sface_to_field,mface_to_sface)
    tface_to_field_s = lazy_map(MyReindex(mface_to_field),tface_to_mface)
    return similar_cell_field(a,tface_to_field_s,ttrian,ReferenceDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  change_domain_ref_ref(a,ttrian,sglue,tglue)
end

function CellData.change_domain(
  a::CellDof,
  strian::BodyFittedTriangulation,::ReferenceDomain,
  ttrian::BodyFittedTriangulation{D,Dp,A,<:ChildUnstructuredGrid,C},::ReferenceDomain
  ) where {D,Dp,A,C}
  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  if strian === ttrian
    return a
  elseif _is_parent(strian,ttrian)
    return CellDof(ttrian.tface_to_mface,ttrian,ReferenceDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  change_domain_ref_ref(a,ttrian,sglue,tglue)
end

dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
idx = [1,10,20]
Ξ = myview(Ω,idx)
dΞ = Measure(Ξ,2)

μ = rand(3)
t = dt
ff = a(μ,t)*∇(dv)⋅∇(du)
trian_f = get_triangulation(ff)

quad = dΩ.quad
@time b = change_domain(ff,quad.trian,quad.data_domain_style)
@time cell_map = get_cell_map(quad.trian)
@time cell_Jt = lazy_map(∇,cell_map)
@time cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
x = get_cell_points(quad)
@time bx = b(x)
@time integral = lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)

_quad = dΞ.quad
@time _b = change_domain(ff,_quad.trian,_quad.data_domain_style)
@time _cell_map = get_cell_map(_quad.trian)
@time _cell_Jt = lazy_map(∇,_cell_map)
@time _cell_Jtx = lazy_map(evaluate,_cell_Jt,_quad.cell_point)
_x = get_cell_points(_quad)
@time _bx = _b(_x)
@time _integral = lazy_map(IntegrationMap(),_bx,_quad.cell_weight,_cell_Jtx)

@time ∫(a(μ,t)*∇(dv)⋅∇(du))dΩ
@time ∫(a(μ,t)*∇(dv)⋅∇(du))dΞ

params,times = realization(feop,10),get_times(fesolver)
@time ∫(aμt(params,times)*∇(dv)⋅∇(du))dΩ
@time ∫(aμt(params,times)*∇(dv)⋅∇(du))dΞ

Ωhat = view(Ω,idx)
dΩhat = Measure(Ωhat,2)
@time ∫(aμt(params,times)*∇(dv)⋅∇(du))dΩhat

_m = DiscreteModelPortion(model,idx)
_t = Triangulation(_m)
_dt = Measure(_t,2)
_test = TestFESpace(_m,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
_trial = PTTrialFESpace(_test,g)
_dv = get_fe_basis(_test)
_du = get_trial_fe_basis(_trial(nothing,nothing))
@time ∫(aμt(params,times)*∇(_dv)⋅∇(_du))_dt

_Ωhat = ReducedTriangulation(Ω,idx)
_dΩhat = Measure(Ωhat,2)
@time ∫(aμt(params,times)*∇(dv)⋅∇(du))_dΩhat

sglue = get_glue(Ω,Val(2))
mface_to_sface = sglue.mface_to_tface
tface_to_mface = Ξ.tface_to_mface

function _lazy_map(k::MyReindex{<:PTArray},j_to_i::AbstractArray) where T
  map(value -> lazy_map(MyReindex(value),j_to_i),k.values)
end

temp = aμt(params,times)*∇(dv)
j = temp.args[1]
sface_to_field = get_data(j)
mface_to_field = extend(sface_to_field,mface_to_sface)
tface_to_field_s = _lazy_map(MyReindex(mface_to_field),tface_to_mface)

ff = a(μ,t)*∇(dv)
jok = ff.args[1]
sface_to_fieldok = get_data(jok)
mface_to_fieldok = extend(sface_to_fieldok,mface_to_sface)
tface_to_field_sok = lazy_map(MyReindex(mface_to_fieldok),tface_to_mface)

@assert typeof(testitem(tface_to_field_s)) == typeof(tface_to_field_sok)


k = MyReindex(mface_to_field)
fi = map(testitem,(tface_to_mface,))
T = return_type(k,fi...)
lazy_map(k,T,tface_to_mface)
item = lazy_map(k,T,tface_to_mface)
item = map(value -> lazy_map(value,T,tface_to_mface),k.values)

_quad = dΞ.quad
_b = change_domain(temp,_quad.trian,_quad.data_domain_style)
_cell_map = get_cell_map(_quad.trian)
_cell_Jt = lazy_map(∇,_cell_map)
_cell_Jtx = lazy_map(evaluate,_cell_Jt,_quad.cell_point)
_x = get_cell_points(_quad)
_bx = _b(_x)
_integral = lazy_map(IntegrationMap(),_bx,_quad.cell_weight,_cell_Jtx)

Γnhat = view(Γn,[1,5,10])
dΓnhat = Measure(Γnhat,2)
@time dc = ∫(hμt(params,times)*dv)dΓn
@time dchat = ∫(hμt(params,times)*dv)dΓnhat

_Γnhat = ReducedTriangulation(Γn,[1,5,10])
_dΓnhat = Measure(_Γnhat,2)
@time ∫(hμt(params,times)*dv)_dΓnhat
rtrian = ReducedTriangulation(Γn.trian,[1,5,10])

# rtrian = ReducedBodyFittedTriangulation(parent_id,model,boundary_grid,Γn.trian.tface_to_mface)
rtrian = _ReducedTriangulation(Γn,[1,5,10])
boh = Measure(rtrian,2)
@time dcboh = ∫(hμt(params,times)*dv)boh

dchat[Γnhat] == dcboh[rtrian]

quad = boh.quad #dΓn.quad
_f = hμt(params,times)*dv
trian_f = get_triangulation(_f)
trian_x = get_triangulation(quad)
is_change_possible(trian_f,trian_x)

b = change_domain(f,quad.trian,quad.data_domain_style)
x = get_cell_points(quad)
bx = b(x)
cell_map = get_cell_map(quad.trian)
cell_Jt = lazy_map(∇,cell_map)
cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
