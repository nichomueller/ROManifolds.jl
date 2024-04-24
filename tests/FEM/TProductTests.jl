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

_get_values(a::Fields.LinearCombinationFieldVector) = a.values
ϕ = _get_values(tpshapes)
ψ = TProduct.FieldFactors(map(_get_values,factors),indices_map,Isotropic())
@assert ψ[shapes.shapes_indices_map] == ϕ

tpvx = tpv(tpx)
vx = v(x)

v1 = get_data(v)[1]
x1 = get_data(x)[1]
v1x1 = evaluate(v1,x1)

tpv1 = get_data(tpv)[1]
tpx1 = get_data(tpx)[1]
tpv1x1 = evaluate(tpv1,tpx1)

#evaluate shapefun at x
trian = Triangulation(model)
degree = TProduct.order_2_degree(order)
quad = Quadrature(QUAD,tensor_product,degree)
cell_quad = CellQuadrature(trian,degree)
x = get_cell_points(cell_quad)
reffe = ReferenceFE(QUAD,lagrangian,Float64,order)
shapes = get_shapefuns(reffe)

C = evaluate(reffe.reffe.dofs,reffe.reffe.prebasis)
px = evaluate(reffe.reffe.prebasis,reffe.reffe.dofs.nodes)

_reffe = ReferenceFE(QUAD,TProduct.tplagrangian,T,order)
_C = evaluate(_reffe.dof_basis,_reffe.prebasis)

using OneHotArrays
using Kronecker
K = OneHotMatrix([1,2,5,3,4,6,7,8,9],9)
C ≈ Matrix(K)*kronecker(get_factors(_C)...)

_x = _reffe.dof_basis.nodes
_px = get_factors(evaluate(_reffe.prebasis,_x))

px ≈ K*kronecker(_px...)

shapex = evaluate(reffe.reffe.shapefuns,reffe.reffe.dofs.nodes)
_shapex = kronecker(map(inv,get_factors(_C))...)*kronecker(_px...)
shapex ≈  _shapex

# v(x) vs tpv(tpx)
result = tpv(tpx)[1]
shapes2d = tpv.cell_basis[1]
nodes2d = tpx.cell_ref_point[1]
invchange2d = shapes2d.values
change2d = inv(invchange2d)
bx2d = evaluate(shapes2d.fields,nodes2d)
@check result ≈ bx2d*invchange2d

shapes = get_data(v)[1]
nodes = get_data(x)[1]
bx = evaluate(prebasis,nodes)
change = evaluate(dof_basis,prebasis)
@check bx ≈ bx2d
@check kronecker(bx.factors...) ≈ bx2d
@check change ≈ change2d
@check change.indices_map.rmatrix*kronecker(change.factors...) ≈ change2d
@check inv(kronecker(change.factors...))*inv(change.indices_map.rmatrix) ≈ invchange2d
@check result ≈ bx*inv(kronecker(change.factors...))*inv(change.indices_map.rmatrix)
# v1x1 = evaluate(v1,x1)

newpreb = TensorProductMonomialBasis(prebasis.factors,prebasis.basis,nodes.indices_map,prebasis.isotropy)
newbx = evaluate(newpreb,nodes)
@check newbx ≈ bx2d
