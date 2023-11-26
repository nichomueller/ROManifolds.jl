# struct MyGridView{Dc,Dp,A,B} <: Grid{Dc,Dp}
#   parent::A
#   cell_to_parent_cell::B
#   function MyGridView(parent::Grid,cell_to_parent_cell::AbstractArray)
#     Dc = num_cell_dims(parent)
#     Dp = num_point_dims(parent)
#     A = typeof(parent)
#     B = typeof(cell_to_parent_cell)
#     new{Dc,Dp,A,B}(parent,cell_to_parent_cell)
#   end
# end

# struct EvalAt{A,B}
#   parent::A
#   cell_to_parent_cell::B
# end

# myview(a::Grid,b::AbstractArray) = MyGridView(a,b)

# function myview(t::BodyFittedTriangulation,ids::AbstractArray)
#   model = t.model
#   grid = myview(t.grid,ids)
#   tface_to_mface = lazy_map(Reindex(t.tface_to_mface),ids)# t.tface_to_mface[ids]
#   BodyFittedTriangulation(model,grid,tface_to_mface)
# end

# function MyGridView(parent::Grid,parent_cell_to_mask::AbstractArray{Bool})
#   cell_to_parent_cell = findall(collect1d(parent_cell_to_mask))
#   MyGridView(parent,cell_to_parent_cell)
# end

# function MyGridView(parent::Grid,parent_cell_to_mask::AbstractVector{Bool})
#   cell_to_parent_cell = findall(parent_cell_to_mask)
#   MyGridView(parent,cell_to_parent_cell)
# end

# function Geometry.OrientationStyle(::Type{MyGridView{Dc,Dp,G}}) where {Dc,Dp,G}
#   OrientationStyle(G)
# end

# function Geometry.RegularityStyle(::Type{MyGridView{Dc,Dp,G}}) where {Dc,Dp,G}
#   RegularityStyle(G)
# end

# function Geometry.get_node_coordinates(grid::MyGridView)
#   get_node_coordinates(grid.parent)
# end

# function Geometry.get_cell_node_ids(grid::MyGridView)
#   get_cell_node_ids(grid.parent)[grid.cell_to_parent_cell]
# end

# function Geometry.get_reffes(grid::MyGridView)
#   get_reffes(grid.parent)
# end

# function Geometry.get_cell_type(grid::MyGridView)
#   get_cell_type(grid.parent)[grid.cell_to_parent_cell]
# end

# function Geometry.get_cell_map(grid::MyGridView)
#   # EvalAt(get_cell_map(grid.parent),grid.cell_to_parent_cell)
#   lazy_map(Reindex(get_cell_map(grid.parent)),grid.cell_to_parent_cell)
# end

# function Geometry.get_facet_normal(grid::MyGridView)
#   # EvalAt(get_facet_normal(grid.parent),grid.cell_to_parent_cell)
#   lazy_map(Reindex(get_facet_normal(grid.parent)),grid.cell_to_parent_cell)
# end

struct MaskedCellPoint{DS,A,B,C} <: CellDatum
  cell_point::CellPoint{DS,A,B,C}
  mask::Vector{Int}
end

CellData.get_cell_points(x::MaskedCellPoint) = x.cell_point
get_mask(x::MaskedCellPoint) = x.mask
get_triangulation(x::MaskedCellPoint) = get_triangulation(get_cell_points(x))
DomainStyle(x::MaskedCellPoint) = DomainStyle(get_cell_points(x))

function get_data(x::MaskedCellPoint)
  mask = get_mask(x)
  function _get_data(data::CompressedArray)
    CompressedArray(data[mask],data.ptrs[mask])
  end
  function _get_data(data::LazyArray)
    lazy_map(Reindex(data),mask)
  end
  _get_data(get_data(get_cell_points(x)))
end

function change_domain(a::MaskedCellPoint,args...)
  MaskedCellPoint(change_domain(get_cell_points(a),args...),get_mask(a))
end

function get_masked_cell_points(trian::Triangulation,mask::Vector{Int})
  MaskedCellPoint(get_cell_points(trian),mask)
end

function get_masked_cell_points(a::CellQuadrature,mask::Vector{Int})
  MaskedCellPoint(get_cell_points(a),mask)
end

function CellData.evaluate!(cache,f::CellField,x::MaskedCellPoint)
  _f, _x = CellData._to_common_domain(f,x)
  cell_field = get_data(_f)
  cell_point = get_data(_x)
  lazy_map(evaluate,cell_field,cell_point)
end

function CellData._to_common_domain(f::CellField,x::MaskedCellPoint)
  trian_x = get_triangulation(x)
  f_on_trian_x = change_domain(f,trian_x,DomainStyle(x))
  f_on_trian_x, x
end

function evaluate!(cache,f::OperationCellField,x::MaskedCellPoint)
  ax = map(i->i(x),f.args)
  lazy_map(Fields.BroadcastingFieldOpMap(f.op.op),ax...)
end

μ = rand(3)
t = dt
dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
idx = [1,10,20]
Ξ = view(Ω,idx)
dΞ = Measure(Ξ,2)

@time dc = ∫(a(μ,t)*∇(dv)⋅∇(du))dΩ
@time _dchat = ∫(a(μ,t)*∇(dv)⋅∇(du))dΞ

∫(a(μ,t)*∇(dv)⋅∇(du))dΞ

ff = a(μ,t)*∇(dv)⋅∇(du)
quad = dΩ.quad
trian_f = get_triangulation(ff)
trian_x = get_triangulation(quad)
b = change_domain(ff,quad.trian,quad.data_domain_style)
cell_map = get_cell_map(quad.trian)
cell_Jt = lazy_map(∇,cell_map)
cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)

x = get_cell_points(quad)
@time bx = b(x)
@time lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)

x̃ = get_masked_cell_points(quad,idx)
@time bx̃ = b(x̃)
@time lazy_map(IntegrationMap(),bx̃,quad.cell_weight,cell_Jtx)

cell_field = get_data(∇(du))
cell_point = get_data(x)
_cell_point = get_data(x̃)
@time lazy_map(evaluate,cell_field,cell_point)
@time lazy_map(evaluate,cell_field,_cell_point)

@time i_to_basis_x = lazy_map(evaluate,cell_field.args[1],cell_point)
@time _i_to_basis_x = lazy_map(evaluate,cell_field.args[1],_cell_point)

@time lazy_map(Gridap.Fields.TransposeMap(),i_to_basis_x)
@time lazy_map(Gridap.Fields.TransposeMap(),_i_to_basis_x)

fi = map(testitem,(i_to_basis_x,))
T = return_type(Gridap.Fields.TransposeMap(),i_to_basis_x)
lazy_map(Gridap.Fields.TransposeMap(),T,i_to_basis_x)

_fi = map(testitem,(_i_to_basis_x,))
_T = return_type(Gridap.Fields.TransposeMap(),_i_to_basis_x)
lazy_map(Gridap.Fields.TransposeMap(),_T,_i_to_basis_x)

# WHY IS THE TYPE DIFFERENT
# _i_to_basis_x = lazy_map(evaluate,cell_field.args[1],_cell_point)
_a = cell_field.args[1]
function rettype(k,f::AbstractArray...)
  fi = map(testitem,f)
  return_type(k,fi...)
end

x = cell_point
fx = map(fi->lazy_map(evaluate,fi,x),_a.args)
# lazy_map(evaluate,_a.args[2],x)
T = rettype(evaluate,_a.args[2],x)
lazy_map(evaluate,_T,_a.args[2],x)

_x = _cell_point
_fx = map(fi->lazy_map(evaluate,fi,_x),_a.args)
# lazy_map(evaluate,_a.args[2],_x)
_T = rettype(evaluate,_a.args[2],_x)
lazy_map(evaluate,_T,_a.args[2],_x)

# function lazy_map(::typeof(evaluate),::Type{T},g::CompressedArray...) where T
#   if _have_same_ptrs(g)
#     _lazy_map_compressed(g...)
#   else
#     LazyArray(T,g...)
#   end
# end

Arrays._have_same_ptrs((_a.args[2],x))
Arrays._have_same_ptrs((_a.args[2],_x))

#
ddu = ∇(du)
get_data(∇(du))


# struct MyReindex{A} <: Map
#   values::A
# end

# function Arrays.testargs(k::MyReindex,i)
#   @check length(k.values) !=0 "This map has empty domain"
#   (one(i),)
# end
# function Arrays.testargs(k::MyReindex,i::Integer...)
#   @check length(k.values) !=0 "This map has empty domain"
#   map(one,i)
# end
# function return_value(k::MyReindex,i...)
#   length(k.values)!=0 ? evaluate(k,testargs(k,i...)...) : testitem(k.values)
# end
# Arrays.return_cache(k::MyReindex,i...) = array_cache(k.values)
# Arrays.evaluate!(cache,k::MyReindex,i...) = getindex!(cache,k.values,i...)
# Arrays.evaluate(k::MyReindex,i...) = k.values[i...]

# function lazy_map(k::MyReindex{<:Fill},::Type{T}, j_to_i::AbstractArray) where T
#   v = k.values.value
#   Fill(v,size(j_to_i)...)
# end

# function lazy_map(k::MyReindex{<:CompressedArray},::Type{T}, j_to_i::AbstractArray) where T
#   i_to_v = k.values
#   values = i_to_v.values
#   ptrs = i_to_v.ptrs[j_to_i]
#   CompressedArray(values,ptrs)
# end

# function lazy_map(k::MyReindex{<:LazyArray},::Type{T},j_to_i::AbstractArray) where T
#   i_to_maps = k.values.maps
#   i_to_args = k.values.args
#   j_to_maps = lazy_map(MyReindex(i_to_maps),eltype(i_to_maps),j_to_i)
#   j_to_args = map(i_to_fk->lazy_map(MyReindex(i_to_fk),eltype(i_to_fk),j_to_i),i_to_args)
#   LazyArray(T,j_to_maps,j_to_args...)
# end

# function lazy_map(k::MyReindex{<:Fields.MemoArray},::Type{T},j_to_i::AbstractArray) where T
#   i_to_v = k.values.parent
#   j_to_v = lazy_map(MyReindex(i_to_v),T,j_to_i)
#   MemoArray(j_to_v)
# end

# struct ValuesAtMask{A,B} <: Map
#   values::A
#   mask::B
# end

# function return_value(k::ValuesAtMask,i...)
#   length(k.values)!=0 ? evaluate(k,testargs(k,i...)...) : testitem(k.values)
# end

# Arrays.return_cache(k::ValuesAtMask,i...) = array_cache(k.values)
# Arrays.evaluate!(cache,k::ValuesAtMask,i...) = getindex!(cache,k.values,k.mask[i...])
# Arrays.evaluate(k::ValuesAtMask,i...) = k.values[k.mask[i...]]

# function lazy_map(k::ValuesAtMask{<:Fill},::Type{T}) where T
#   v = k.values.value
#   Fill(v,size(k.mask)...)
# end

# function lazy_map(k::ValuesAtMask{<:CompressedArray},::Type{T}) where T
#   i_to_v = k.values
#   values = i_to_v.values
#   ptrs = i_to_v.ptrs[k.mask]
#   CompressedArray(values,ptrs)
# end

# function lazy_map(k::ValuesAtMask{<:LazyArray},::Type{T}) where T
#   i_to_maps = k.values.maps
#   i_to_args = k.values.args
#   j_to_maps = lazy_map(ValuesAtMask(i_to_maps,k.mask),eltype(i_to_maps))
#   j_to_args = map(i_to_fk->lazy_map(ValuesAtMask(i_to_fk,k.mask),eltype(i_to_fk)),i_to_args)
#   LazyArray(T,j_to_maps,j_to_args...)
# end

# function lazy_map(f,k::ValuesAtMask,args...)
#   ValuesAtMask(lazy_map(f,k.values,args...),k.mask)
# end

# myview(a::Grid,b::AbstractArray) = MyGridView(a,b)

# function myview(t::BodyFittedTriangulation,ids::AbstractArray)
#   model = t.model
#   grid = myview(t.grid,ids)
#   tface_to_mface = t.tface_to_mface[ids] # lazy_map(MyReindex(t.tface_to_mface),ids) #
#   BodyFittedTriangulation(model,grid,tface_to_mface)
# end

# function MyGridView(parent::Grid,parent_cell_to_mask::AbstractArray{Bool})
#   cell_to_parent_cell = findall(collect1d(parent_cell_to_mask))
#   MyGridView(parent,cell_to_parent_cell)
# end

# function MyGridView(parent::Grid,parent_cell_to_mask::AbstractVector{Bool})
#   cell_to_parent_cell = findall(parent_cell_to_mask)
#   MyGridView(parent,cell_to_parent_cell)
# end

# function Geometry.OrientationStyle(::Type{MyGridView{Dc,Dp,G}}) where {Dc,Dp,G}
#   OrientationStyle(G)
# end

# function Geometry.RegularityStyle(::Type{MyGridView{Dc,Dp,G}}) where {Dc,Dp,G}
#   RegularityStyle(G)
# end

# function Geometry.get_node_coordinates(grid::MyGridView)
#   get_node_coordinates(grid.parent)
# end

# function Geometry.get_cell_node_ids(grid::MyGridView)
#   # get_cell_node_ids(grid.parent)[grid.cell_to_parent_cell]
#   ValuesAtMask(get_cell_node_ids(grid.parent),grid.cell_to_parent_cell)
# end

# function Geometry.get_reffes(grid::MyGridView)
#   get_reffes(grid.parent)
# end

# function Geometry.get_cell_type(grid::MyGridView)
#   # get_cell_type(grid.parent)[grid.cell_to_parent_cell]
#   ValuesAtMask(get_cell_type(grid.parent),grid.cell_to_parent_cell)
# end

# function Geometry.get_cell_map(grid::MyGridView)
#   ValuesAtMask(get_cell_map(grid.parent),grid.cell_to_parent_cell)
#   # lazy_map(MyReindex(get_cell_map(grid.parent)),grid.cell_to_parent_cell)
# end

# function Geometry.get_facet_normal(grid::MyGridView)
#   ValuesAtMask(get_facet_normal(grid.parent),grid.cell_to_parent_cell)
#   # lazy_map(MyReindex(get_facet_normal(grid.parent)),grid.cell_to_parent_cell)
# end
