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
add_tag_from_tags!(labels,"neumann",[7])

D = 2
T = VectorValue{2,Float64} #Float64#
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
nodes_map = compute_nodes_map(;polytope=QUAD,orders=(2,2))
nodes = get_nodes(dof_basis)
_nodes = TensorProductNodes(QUAD,2)

ξ = evaluate(prebasis,dof_basis.nodes)
ϕξ = evaluate(dof_basis,prebasis)

_prebasis = TensorProductMonomialBasis(T,QUAD,2)
_dof_basis = TensorProductDofBases(T,QUAD,2)
_ξ = evaluate(_prebasis,_nodes)
_ϕξ = evaluate(_dof_basis,_prebasis)

tpξ = evaluate(tpprebasis,tpdof_basis.nodes)
tpϕξ = evaluate(tpdof_basis,tpprebasis)

M = tpϕξ[1:9,1:2:18]
m = _ϕξ[1][1:3,1:2:6]

struct Dof2NodeAndComp{A,B}
  node_map::A
  dofs_map::B
end

num_nodes(a::Dof2NodeAndComp) = length(a.node_map)

struct Temp{T,A,B} <: AbstractMatrix{T}
  a::Vector{Matrix{T}}
  i::Dof2NodeAndComp{A,B}
  size::NTuple{2,Int}
end

Base.size(a::Temp) = a.size
Base.IndexStyle(::Temp) = IndexCartesian()

function Base.getindex(a::Temp,i::Integer,j::Integer)
  nnodes = num_nodes(a.i)
  ncomps = 2
  compi = slow_index(i,nnodes)
  compj = fast_index(j,ncomps)
  if compi != compj
    return zero(eltype(a))
  end
  nodei = fast_index(i,nnodes)
  nodej = slow_index(j,ncomps)
  rowi = a.i.node_map[nodei]
  colj = a.i.dofs_map[nodej]
  return prod(map(d->a.a[d][rowi[d],colj[d]],1:D))
end

node_map = _nodes.nodes_map
dofs_map = collect1d(CartesianIndices((1:3,1:3)))
mymap = Dof2NodeAndComp(node_map,dofs_map)
temp = Temp(collect(_ϕξ),mymap,(18,18))

###########################
a = temp
i,j = 10,2
nnodes = num_nodes(a.i)
ncomps = 2
compi = slow_index(i,nnodes)
compj = fast_index(j,ncomps)
if compi != compj
  return zero(eltype(a))
end
nodei = fast_index(i,nnodes)
nodej = slow_index(j,nnodes)
rowi = a.i.node_map[nodei]
colj = a.i.dofs_map[nodej]
###########################

B = zeros(size(A))
for i = axes(B,1)
  entry = node_map[i]
  B[i,:] = kronecker(_ξ[2][entry[2],:],_ξ[1][entry[1],:])
end

BB = zeros(size(A))
for irow = axes(BB,1)
  entryrow = node_map[irow]
  factors = map(d->_ξ[D-d+1][entryrow[D-d+1],:],1:D)
  BB[irow,:] = kronecker(factors...)
end
