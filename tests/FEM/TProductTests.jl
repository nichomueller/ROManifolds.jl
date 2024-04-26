using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.Helpers
using BlockArrays
using FillArrays

using Mabla.FEM
using Mabla.FEM.TProduct

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,3,7]) #1,2,3,4,5,6,8

T = Float64
order = 1
reffe = ReferenceFE(QUAD,lagrangian,T,order)
cell_reffe = Fill(reffe,num_cells(model))
conformity=nothing
trian = Triangulation(model)
labels = get_face_labeling(model)
dirichlet_tags=["dirichlet"]
dirichlet_masks=nothing
constraint=nothing
vector_type=nothing
conf = Conformity(testitem(cell_reffe),conformity)
vector_type = Vector{Float64}
cell_fe = CellFE(model,cell_reffe,conf)

grid_topology = get_grid_topology(model)
ntags = length(dirichlet_tags)

cell_to_ctype = cell_fe.cell_ctype
ctype_to_ldof_to_comp = cell_fe.ctype_ldof_comp
ctype_to_num_dofs = cell_fe.ctype_num_dofs

cell_conformity = CellConformity(cell_fe)
d_to_ctype_to_ldface_to_own_ldofs = cell_conformity.d_ctype_ldface_own_ldofs
ctype_to_lface_to_own_ldofs = cell_conformity.ctype_lface_own_ldofs
ctype_to_lface_to_pindex_to_pdofs = cell_conformity.ctype_lface_pindex_pdofs

D = num_cell_dims(grid_topology)
n_faces = num_faces(grid_topology)
d_to_cell_to_dfaces = [ Table(get_faces(grid_topology,D,d)) for d in 0:D]
d_to_dface_to_cells = [ Table(get_faces(grid_topology,d,D)) for d in 0:D]
d_to_offset = get_offsets(grid_topology)

face_to_own_dofs, ntotal, d_to_dface_to_cell, d_to_dface_to_ldface = FESpaces._generate_face_to_own_dofs(
    n_faces,
    cell_to_ctype,
    d_to_cell_to_dfaces,
    d_to_dface_to_cells,
    d_to_offset,
    d_to_ctype_to_ldface_to_own_ldofs)

d_to_dface_to_tag = [ get_face_tag_index(labels,dirichlet_tags,d)  for d in 0:D]
cell_to_faces = Table(get_cell_faces(grid_topology))

# face_to_entity = get_face_entity(labels,d)
# tag_to_entities = get_tag_entities(labels)

_dirichlet_components = FESpaces._convert_dirichlet_components(dirichlet_tags,dirichlet_masks)

nfree, ndiri, diri_dof_tag = FESpaces._split_face_own_dofs_into_free_and_dirichlet!(
  face_to_own_dofs,
  d_to_offset,
  d_to_dface_to_tag,
  d_to_dface_to_cell,
  d_to_dface_to_ldface,
  cell_to_ctype,
  d_to_ctype_to_ldface_to_own_ldofs,
  ctype_to_ldof_to_comp,
  _dirichlet_components)

cell_to_lface_to_pindex = Table(get_cell_permutations(grid_topology))

cell_dofs = FESpaces.CellDofsNonOriented(
  cell_to_faces,
  cell_to_lface_to_pindex,
  cell_to_ctype,
  ctype_to_lface_to_own_ldofs,
  ctype_to_num_dofs,
  face_to_own_dofs,
  ctype_to_lface_to_pindex_to_pdofs)

a = cell_dofs
cell = 5
ctype = a.cell_to_ctype[cell]
n_dofs = a.ctype_to_num_dofs[ctype]
lface_to_own_ldofs = a.ctype_to_lface_to_own_ldofs[ctype]
p = a.cell_to_faces.ptrs[cell]-1
# for (lface, own_ldofs) in enumerate(lface_to_own_ldofs)
lface, own_ldofs = 1, lface_to_own_ldofs
  face = a.cell_to_faces.data[p+lface]
  pindex = a.cell_to_lface_to_pindex.data[p+lface]
  pdofs = a.ctype_to_lface_to_pindex_to_pdofs[ctype][lface][pindex]

  q = a.face_to_own_dofs.ptrs[face]-1
  for (i,ldof) in enumerate(own_ldofs)
    j = pdofs[i]
    dof = a.face_to_own_dofs.data[q+j]
    dofs[ldof] = dof
  end
# end

D = 2
T = Float64
order = 2

orders = (2,2)
prebasis = TensorProductMonomialBasis(T,QUAD,orders)
dof_basis = TensorProductDofBases(T,QUAD,lagrangian,orders)
pd = evaluate(dof_basis,prebasis)

reffe = ReferenceFE(QUAD,TProduct.tplagrangian,T,order)
shapes = get_shapefuns(reffe)
dof_basis = get_dof_basis(reffe)
prebasis = get_prebasis(reffe)
space = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
v = get_fe_basis(space)

tpreffe = ReferenceFE(QUAD,lagrangian,T,order)
tpshapes = get_shapefuns(tpreffe)
tpdof_basis = get_dof_basis(tpreffe)
tpprebasis = get_prebasis(tpreffe)
tpspace = TestFESpace(model,tpreffe;conformity=:H1,dirichlet_tags=["dirichlet"])
tpv = get_fe_basis(tpspace)

evaluate(tpdof_basis,tpshapes)

tpϕx = evaluate(tpprebasis,tpdof_basis.nodes)
ϕx = evaluate(prebasis,dof_basis.nodes)
@assert tpϕx ≈ ϕx

bx = evaluate(dof_basis,prebasis)
tpbx = evaluate(tpdof_basis,tpprebasis)
@assert bx ≈ tpbx

trian = Triangulation(model)

degree = TProduct.order_2_degree(order)
tpquad = Quadrature(QUAD,degree)
quad = Quadrature(QUAD,tpquadrature,degree)

cell_quad = CellQuadrature(trian,degree)
# _cell_quad = CellQuadrature(trian,quad)
tpn = TensorProductNodes(map(get_coordinates,quad.factors),quad.quad_map,TProduct.Isotropic())
x = CellPoint(Fill(tpn,4),trian,ReferenceDomain())

tpx = get_cell_points(cell_quad)
@assert all(v(tpx) .≈ tpv(tpx))
@assert all(tpv(x) .≈ tpv(tpx))
@assert all(v(x) .≈ tpv(tpx))

############ Geometry
grid = TensorProductGrid(domain,partition)
nodes = get_node_coordinates(grid)
desc = get_cartesian_descriptor(grid)
amap = get_cell_map(grid)
############ Cell fields

f(x) = exp(x[1])

cf = f*v
tpcf = f*tpv
tpcfx = tpcf(x)
cfx = cf(x)
@check all(cfx .≈ tpcfx)

_f,_x = CellData._to_common_domain(cf,x)
cell_field = get_data(_f)
cell_point = get_data(_x)
X = lazy_map(evaluate,cell_field,cell_point)

# v = return_value(cell_field[1],cell_point[1])
evaluate(cell_field[1],testargs(cell_field[1],cell_point[1])...)
c = return_cache(cell_field[1],testargs(cell_field[1],cell_point[1])...)

ff = cell_field[1]
xx = cell_point[1]
cfs = map(fi -> return_cache(fi,xx),ff.args)
rs = map(fi -> return_value(fi,xx),ff.args)
bm = Fields.BroadcastingFieldOpMap(ff.op)
r = return_cache(bm,rs...)






BOH
# cf = v*v
# tpcf = tpv*tpv
# tpcfx = tpcf(x)
# cfx = cf(x)
# @check all(cfx .≈ tpcfx)

# cf = ∇(v)⋅∇(v)
# tpcf = ∇(tpv)⋅∇(tpv)
# tpcfx = tpcf(x)
# cfx = cf(x)
# @check all(cfx .≈ tpcfx)
ff,xx = cell_field[1],cell_point[1]
cfs = map(fi -> return_cache(fi,xx),ff.args)
rs = map(fi -> return_value(fi,xx),ff.args)
bm = Fields.BroadcastingFieldOpMap(ff.op)
r = return_cache(bm,rs...)
r, cfs

return_value(ff.args[2],xx)
testargs(ff.args[2],xx)
