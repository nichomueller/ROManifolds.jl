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
nodes_map = compute_indices_map(QUAD,(2,2))
nodes = get_nodes(dof_basis)
_nodes = TensorProductNodes(QUAD,2)

ξ = evaluate(prebasis,dof_basis.nodes)
ϕξ = evaluate(dof_basis,prebasis)

_prebasis = TensorProductMonomialBasis(Float64,QUAD,2)
_dof_basis = TensorProductDofBases(Float64,QUAD,2)
_ξ = evaluate(_prebasis,_nodes)
_ϕξ = evaluate(_dof_basis,_prebasis)

cache = return_cache(_dof_basis,_prebasis)

tpξ = evaluate(tpprebasis,tpdof_basis.nodes)
tpϕξ = evaluate(tpdof_basis,tpprebasis)

M = tpϕξ[1:9,1:2:18]
m = _ϕξ[1][1:3,1:2:6]

M ≈ kronecker(m,m)

nnodes1 = nnodes2 = 6
nnodes = 9
ncomps = 2
ndofs = nnodes*ncomps

node = 1
comp = 1
col = 2

node1,node2 = Tuple(nodes_map[node])

dof = tpdof_basis.node_and_comp_to_dof[node][comp]
dof1 = dof_basis.node_and_comp_to_dof[node1][comp]
dof2 = dof_basis.node_and_comp_to_dof[node2][comp]

comp_col = mod(col-1,ncomps) + 1
node_col = Int(floor((col-1)/ncomps)+1)
node1_col,node2_col = Tuple(nodes_map[node_col])
col1 = (comp_col-1)*nnodes_loc + node1_col
col2 = (comp_col-1)*nnodes_loc + node2_col

_ϕξ[1][dof1,col1]*_ϕξ[2][dof2,col2]

_ξ[1][node1,1][comp]
_ξ[2][node2,1][comp]

nnodes_loc = 3

for node = 1:nnodes
  node1,node2 = Tuple(nodes_map[node])
  for col = 1:ndofs
    for comp = 1:ncomps
      dof = tpdof_basis.node_and_comp_to_dof[node][comp]
      dof1 = dof_basis.node_and_comp_to_dof[node1][comp]
      dof2 = dof_basis.node_and_comp_to_dof[node2][comp]

      # col1 = (node1-1)*ncomps + comp
      # col2 = (node2-1)*ncomps + comp
      # col1 = (comp-1)*nnodes_loc + node1
      # col2 = (comp-1)*nnodes_loc + node2
      # comp_col = mod(col-1,ncomps) + 1
      # node_col = Int(floor((col-1)/ncomps)+1)
      comp_col = Int(floor((col-1)/nnodes)+1)
      node_col = mod(col-1,nnodes) + 1
      node1_col,node2_col = Tuple(nodes_map[node_col])
      col1 = (comp_col-1)*nnodes_loc + node1_col
      col2 = (comp_col-1)*nnodes_loc + node2_col

      # A[dof,col] = _ϕξ[1][dof1,col1]*_ϕξ[2][dof2,col2]
      A[dof,col] = _ξ[1][node1,col1][comp]*_ξ[2][node2,col2][comp]
    end
  end
end

node = 1
col = 1
comp = 2

node1,node2 = Tuple(nodes_map[node])

dof = tpdof_basis.node_and_comp_to_dof[node][comp]

A = tpϕξ
_A = kronecker(_ξ[1],_ξ[2])
node_map = _nodes.nodes_map


B = zeros(size(A))
for i = axes(B,1)
  entry = node_map[i]
  B[i,:] = kronecker(_ξ[2][entry[2],:],_ξ[1][entry[1],:])
end

struct Dof2NodeAndComp{A,B}
  node_map::A
  dofs_map::B
end

num_nodes(a::Dof2NodeAndComp) = length(a.node_map)

struct Temp1{T,A,B} <: AbstractMatrix{T}
  a::Vector{Matrix{T}}
  i::Dof2NodeAndComp{A,B}
end

Base.size(a::Temp1) = ntuple(d->prod(size.(a.a,d)),length(a.a))
Base.IndexStyle(::Temp1) = IndexCartesian()

function Base.getindex(a::Temp1,i::Integer,j::Integer)
  nnodes = num_nodes(a.i)
  compi = slow_index(i,nnodes)
  compj = slow_index(j,nnodes)
  # compi != compj && return zero(eltype(a))
  if compi != compj
    return zero(eltype(a))
  end
  nodei = fast_index(i,nnodes)
  rowi = a.i.node_map[nodei]
  coli = a.i.dof_map[nodei]
  return prod(map(d->a.a[rowi[d],coli[d]]))
end

mymap = Dof2NodeAndComp(node_map,)
