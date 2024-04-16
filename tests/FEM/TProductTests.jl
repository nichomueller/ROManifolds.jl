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

using Mabla.FEM
using Mabla.FEM.TProduct

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

D = 2
T = Float64#VectorValue{2,Float64}
order = 2

reffe = ReferenceFE(SEGMENT,lagrangian,T,order)
prebasis = get_prebasis(reffe)
dof_basis = get_dof_basis(reffe)
change = inv(evaluate(dof_basis,prebasis))

tpreffe = ReferenceFE(QUAD,lagrangian,T,order)
tpshapes = get_shapefuns(tpreffe)
tpdof_basis = get_dof_basis(tpreffe)
tpprebasis = get_prebasis(tpreffe)

################### TPNodes ##########
nodes_map = indices_map(QUAD,(2,2))
nodes = get_nodes(dof_basis)
tpnodes = TensorProductNodes((nodes,nodes),nodes_map)

ξ = evaluate(prebasis,dof_basis.nodes)

_prebasis = TensorProductMonomialBasis((prebasis,prebasis))
_ξ = evaluate(_prebasis,tpnodes)

tpξ = evaluate(tpprebasis,tpdof_basis.nodes)
