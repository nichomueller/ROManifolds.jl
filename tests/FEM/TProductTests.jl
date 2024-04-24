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
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])

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
