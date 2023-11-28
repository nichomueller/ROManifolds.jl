dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
idx = collect(1:5:50)
μ = rand(3)
t = dt
params = realization(feop,10)
times = get_times(fesolver)

########################### NEW TRIAN STUFF ####################################
struct ReducedUnstructuredGrid{Dc,Dp,Tp,O,Tn} <: Grid{Dc,Dp}
  node_coordinates::Vector{Point{Dp,Tp}}
  cell_node_ids::Table{Int32,Vector{Int32},Vector{Int32}}
  reffes::Vector{LagrangianRefFE{Dc}}
  cell_types::Vector{Int8}
  orientation_style::O
  facet_normal::Tn
  cell_map
  function ReducedUnstructuredGrid(
    parent::UnstructuredGrid{Dc,Dp,Tp,O,Tn},
    cell_to_parent_cell::AbstractArray) where {Dc,Dp,Tp,O,Tn}

    node_coordinates = get_node_coordinates(parent)
    cell_node_ids = Table(get_cell_node_ids(parent)[cell_to_parent_cell])
    reffes = get_reffes(parent)
    cell_types = get_cell_type(parent)[cell_to_parent_cell]
    orientation_style = parent.orientation_style
    facet_normal = if Tn == Nothing
      nothing
    else
      lazy_map(Reindex(get_facet_normal(parent)),cell_to_parent_cell)
    end
    cell_map = lazy_map(Reindex(get_cell_map(parent)),cell_to_parent_cell)
    new{Dc,Dp,Tp,O,Tn}(node_coordinates,cell_node_ids,reffes,cell_types,
      orientation_style,facet_normal,cell_map)
  end
end

Geometry.get_reffes(g::ReducedUnstructuredGrid) = g.reffes
Geometry.get_cell_type(g::ReducedUnstructuredGrid) = g.cell_types
Geometry.get_node_coordinates(g::ReducedUnstructuredGrid) = g.node_coordinates
Geometry.get_cell_node_ids(g::ReducedUnstructuredGrid) = g.cell_node_ids
Geometry.get_cell_map(g::ReducedUnstructuredGrid) = g.cell_map
function Geometry.get_facet_normal(g::ReducedUnstructuredGrid)
  @assert !isnothing(g.facet_normal) "This Grid does not have information about normals."
  g.facet_normal
end

abstract type ReducedTriangulation{Dt,Dp} <: Triangulation{Dt,Dp} end

function ReducedTriangulation(::Triangulation,::AbstractArray)
  @notimplemented
end

function is_parent(parent::Triangulation,child::Triangulation;kwargs...)
  false
end

function Geometry.get_glue(trian::ReducedTriangulation{Dt},::Val{Dt}) where Dt
  tface_to_mface = trian.tface_to_mface
  tface_to_mface_map = Fill(GenericField(identity),num_cells(trian))
  if isa(tface_to_mface,IdentityVector) && num_faces(trian.model,Dt) == num_cells(trian)
    mface_to_tface = tface_to_mface
  else
    nmfaces = num_faces(trian.model,Dt)
    mface_to_tface = PosNegPartition(tface_to_mface,Int32(nmfaces))
  end
  FaceToFaceGlue(tface_to_mface,tface_to_mface_map,mface_to_tface)
end

function change_domain(
  a::CellField,
  strian::Triangulation,::ReferenceDomain,
  ttrian::ReducedTriangulation,::ReferenceDomain)

  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  D = num_cell_dims(strian)
  if is_parent(strian,ttrian)
    sface_to_field = get_data(a)
    sglue = get_glue(strian,Val(D))
    mface_to_sface = sglue.mface_to_tface
    tface_to_mface = ttrian.tface_to_mface
    mface_to_field = extend(sface_to_field,mface_to_sface)
    tface_to_field_s = lazy_map(Reindex(mface_to_field),tface_to_mface)
    return similar_cell_field(a,tface_to_field_s,ttrian,ReferenceDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  CellData.change_domain_ref_ref(a,ttrian,sglue,tglue)
end

function change_domain(
  a::CellDof,
  strian::Triangulation,::ReferenceDomain,
  ttrian::ReducedTriangulation,::ReferenceDomain)

  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  if is_parent(strian,ttrian)
    return CellDof(ttrian.tface_to_mface,ttrian,ReferenceDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  CellData.change_domain_ref_ref(a,ttrian,sglue,tglue)
end

function change_domain(
  a::CellField,
  strian::Triangulation,::PhysicalDomain,
  ttrian::ReducedTriangulation,::PhysicalDomain)

  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  D = num_cell_dims(strian)
  if is_parent(strian,ttrian)
    sface_to_field = get_data(a)
    sglue = get_glue(strian,Val(D))
    mface_to_sface = sglue.mface_to_tface
    tface_to_mface = ttrian.tface_to_mface
    mface_to_field = extend(sface_to_field,mface_to_sface)
    tface_to_field_s = lazy_map(Reindex(mface_to_field),tface_to_mface)
    return similar_cell_field(a,tface_to_field_s,ttrian,PhysicalDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  CellData.change_domain_phys_phys(a,ttrian,sglue,tglue)
end

function change_domain(
  a::CellDof,
  strian::Triangulation,::PhysicalDomain,
  ttrian::ReducedTriangulation,::PhysicalDomain)

  msg = """\n
  We cannot move the given CellField to the reference domain of the requested triangulation.
  Make sure that the given triangulation is either the same as the triangulation on which the
  CellField is defined, or that the latter triangulation is the background of the former.
  """

  if is_parent(strian,ttrian)
    return CellDof(ttrian.tface_to_mface,ttrian,PhysicalDomain())
  end
  @assert is_change_possible(strian,ttrian) msg
  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  CellData.change_domain_phys_phys(a,ttrian,sglue,tglue)
end

struct ReducedBodyFittedTriangulation{Dt,Dp,A,B,C} <: ReducedTriangulation{Dt,Dp}
  parent_id::UInt
  model::A
  grid::B
  tface_to_mface::C
  function ReducedBodyFittedTriangulation(
    parent_id::UInt,model::DiscreteModel,grid::Grid,tface_to_mface)
    Dp = num_point_dims(model)
    @assert Dp == num_point_dims(grid)
    Dt = num_cell_dims(grid)
    A = typeof(model)
    B = typeof(grid)
    C = typeof(tface_to_mface)
    new{Dt,Dp,A,B,C}(parent_id,model,grid,tface_to_mface)
  end
end

function ReducedTriangulation(trian::BodyFittedTriangulation,ids::AbstractArray)
  parent_id = objectid(parent)
  model = trian.model
  grid = ReducedUnstructuredGrid(trian.grid,ids)
  tface_to_mface = trian.tface_to_mface[ids]
  ReducedBodyFittedTriangulation(parent_id,model,grid,tface_to_mface)
end

function ReducedTriangulation(child::ReducedBodyFittedTriangulation,parent::BodyFittedTriangulation)
  parent_id = objectid(parent)
  ReducedBodyFittedTriangulation(parent_id,parent.model,child.grid,child.tface_to_mface)
end

Geometry.get_background_model(trian::ReducedBodyFittedTriangulation) = trian.model
Geometry.get_grid(trian::ReducedBodyFittedTriangulation) = trian.grid

function is_parent(
  parent::BodyFittedTriangulation{Dt,Dp},
  child::ReducedBodyFittedTriangulation{Dt,Dp};
  shallow=false) where {Dt,Dp}

  if shallow
    get_node_coordinates(get_grid(parent)) == get_node_coordinates(get_grid(child))
  else
    objectid(parent) == child.parent_id
  end
end

############################## BODYFITTED ###################################
Ξ = ReducedTriangulation(Ω,idx)
dΞ = Measure(Ξ,2)
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

# new
idx = IdentityVector(10)
Ξ = BodyFittedTriangulation(Ω.model,Ω.grid,idx)
dΞ = Measure(Ξ,2)
ff = a(μ,t)*∇(dv)⋅∇(du)
trian_f = get_triangulation(ff)

_quad = dΞ.quad
@time _b = change_domain(ff,_quad.trian,_quad.data_domain_style)
@time _cell_map = get_cell_map(_quad.trian)
@time _cell_Jt = lazy_map(∇,_cell_map)
@time _cell_Jtx = lazy_map(evaluate,_cell_Jt,_quad.cell_point)
_x = get_cell_points(_quad)
@time _bx = _b(_x)
@time _integral = lazy_map(IntegrationMap(),_bx,_quad.cell_weight,_cell_Jtx)

############################## BOUNDARY ###################################
@time dc = ∫(hμt(params,times)*dv)dΓn

Γnhat = view(Γn,idx)
dΓnhat = Measure(Γnhat,2)
@time dchat = ∫(hμt(params,times)*dv)dΓnhat

_Γnhat = ReducedTriangulation(Γn,idx)
_dΓnhat = Measure(_Γnhat,2)
@time _dchat = ∫(hμt(params,times)*dv)_dΓnhat


############################### NEW TESTS ######################################
ff = a(μ,t)*∇(dv)⋅∇(du)

quad = dΩ.quad
baseline_stats = @timed begin
  b = change_domain(ff,quad.trian,quad.data_domain_style)
  cell_map = get_cell_map(quad.trian)
  cell_Jt = lazy_map(∇,cell_map)
  cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
  x = get_cell_points(quad)
  bx = b(x)
  integral = lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
end

reduced_stats = @timed begin
  b = change_domain(ff,quad.trian,quad.data_domain_style)
  cell_map = get_cell_map(quad.trian)
  cell_Jt = lazy_map(∇,cell_map)
  cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
  x = get_cell_points(quad)
  bx = b(x)
  _bx = lazy_map(Reindex(bx),idx)
  _w = lazy_map(Reindex(quad.cell_weight),idx)
  _Jx = lazy_map(Reindex(cell_Jtx),idx)
  _integral = lazy_map(IntegrationMap(),_bx,_w,_Jx)
end

Ξ = ReducedTriangulation(Ω,idx)
dΞ = Measure(Ξ,2)
_quad = dΞ.quad
prev_red_stats = @timed begin
  _b = change_domain(ff,_quad.trian,_quad.data_domain_style)
  _cell_map = get_cell_map(_quad.trian)
  _cell_Jt = lazy_map(∇,_cell_map)
  _cell_Jtx = lazy_map(evaluate,_cell_Jt,_quad.cell_point)
  _x = get_cell_points(_quad)
  __bx = _b(_x)
  __integral = lazy_map(IntegrationMap(),__bx,_quad.cell_weight,_cell_Jtx)
end

snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
op = get_ptoperator(fesolver,feop,snaps_test,params_test)

form(μ,t,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
opf = form(params_test,times,du,dv)
rdΩ = ReducedMeasure(dΩ,idx)
Ωhat = view(Ω,idx)
dΩhat = Measure(Ωhat,2)
rdc = integrate(opf,rdΩ)#integrate(ff,rdΩ.meas.quad,rdΩ.cell_to_parent_cell)
vdc = integrate(opf.object,dΩhat)
dc = integrate(opf.object,dΩ)

ptrian = get_parent_triangulation(rdΩ)
trian = get_triangulation(rdΩ)

A = allocate_jacobian(op,op.vθ)#[1]
_A = copy(A)
@assert rdc[trian][idx] == dc[Ω][idx]
matdata = collect_cell_matrix(trial(nothing,nothing),test,rdc)
assemble_matrix_add!(A,feop.assem,matdata)

cell_mat,ttrian = move_contributions(rdc[trian],trian)
cell_mat_c = attach_constraints_cols(trial(nothing,nothing),cell_mat,ttrian)
cell_mat_rc = attach_constraints_rows(test,cell_mat_c,ttrian)

_matdata = collect_cell_matrix(trial(nothing,nothing),test,vdc)
assemble_matrix_add!(_A,feop.assem,_matdata)

_cell_mat,_ttrian = move_contributions(vdc[Ωhat],Ωhat)
_cell_mat_c = attach_constraints_cols(trial(nothing,nothing),_cell_mat,_ttrian)
_cell_mat_rc = attach_constraints_rows(test,_cell_mat_c,_ttrian)
