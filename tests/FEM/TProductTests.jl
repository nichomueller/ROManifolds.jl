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
T = VectorValue{2,Float64}
order = 2

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

@assert evaluate(dof_basis,prebasis) ≈ evaluate(tpdof_basis,tpprebasis)

trian = Triangulation(model)

degree = TProduct.order_2_degree(order)
tpquad = Quadrature(QUAD,degree)
quad = Quadrature(QUAD,tpquadrature,degree)

cell_quad = CellQuadrature(trian,degree)
# _cell_quad = CellQuadrature(trian,quad)

x = get_cell_points(cell_quad)
@assert all(v(x) .≈ tpv(x))

tpgrid = CartesianGrid(domain,partition)
affmap = get_cell_map(tpgrid)[1]

tpn = TensorProductNodes(map(get_coordinates,quad.factors),quad.quad_map,TProduct.Isotropic())
x = CellPoint(Fill(tpn,4),trian,ReferenceDomain())

_cache = return_cache(affmap,tpn)
# evaluate!(_cache,affmap,tpn)
factors,indices_map,cache = _cache
points = get_factors(tpn)
r = evaluate!(cache,factors[1],points[1])
tpr = TProduct.FieldFactors(Fill(r,D),indices_map,Isotropic())

_vx = evaluate!(cache,get_data(v),get_data(_x))

_ξ = get_data(_x)[1]
_c = return_cache(ϕ,_ξ)
evaluate!(_c,ϕ,_ξ)

grid = TensorProductGrid(domain,partition)
tpgrid = CartesianGrid(domain,partition)

@assert all(get_node_coordinates(grid) .== get_node_coordinates(tpgrid))
