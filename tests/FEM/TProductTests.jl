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

orders = (2,2)
prebasis = TensorProductMonomialBasis(T,QUAD,orders)
dof_basis = TensorProductDofBases(T,QUAD,lagrangian,orders)
pd = evaluate(dof_basis,prebasis)

tpϕx = evaluate(tpprebasis,tpdof_basis.nodes)

###################
nodes_map = compute_nodes_map(;polytope=QUAD,orders)
ndofs = 18
indices_map = compute_nodes_and_comps_2_dof_map(nodes_map;orders,ndofs)
ϕx = evaluate(prebasis,dof_basis.nodes)
bf = TProduct.FieldFactors(ϕx,indices_map,Isotropic())

nodei,j = 2,3
factors = get_factors(bf)
indices_map = get_indices_map(bf)
ncomps = num_components(indices_map)
compj = FEM.fast_index(j,ncomps)
nodej = FEM.slow_index(j,ncomps)
rowi = indices_map.nodes_map[nodei]
colj = indices_map.dofs_map[nodej]
kk = ntuple(d->factors[d][rowi[d],colj[d]],length(factors))

Base.:*(a::VectorValue,b::VectorValue) = Point(map(*,a.data,b.data))
B = zeros(VectorValue{2,Float64},9,18)
for i = axes(B,1)
  entry = indices_map.nodes_map[i]
  B[i,:] = kronecker(factors[2][entry[2],:],factors[1][entry[1],:])
end
###################

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

tpx = get_cell_points(cell_quad)
@assert all(v(tpx) .≈ tpv(tpx))

tpgrid = CartesianGrid(domain,partition)
affmap = get_cell_map(tpgrid)[1]

tpn = TensorProductNodes(map(get_coordinates,quad.factors),quad.quad_map,TProduct.Isotropic())
x = CellPoint(Fill(tpn,4),trian,ReferenceDomain())
v(x)

f(x) = x

tpvfx = (tpv⋅f)(tpx)
vfx = (v⋅f)(x)

op1 = tpv⋅f
ax = map(i->i(x),op1.args)
# lazy_map(Fields.BroadcastingFieldOpMap(op1.op.op),ax...)
item = ax[1][1],ax[2][1]
k = Fields.BroadcastingFieldOpMap(op1.op.op)
cache = return_cache(k,item...)
eval1 = evaluate!(cache,k,item...)

# function return_cache(f::Broadcasting,x::Union{Number,AbstractArray{<:Number}}...)
#   s = map(_size,x)
#   bs = Base.Broadcast.broadcast_shape(s...)
#   T = return_type(f.f,map(testitem,x)...)
#   r = fill(testvalue(T),bs)
#   cache = CachedArray(r)
#   _prepare_cache!(cache,x...)
#   cache
# end

# function evaluate!(
#   cache,
#   f::BroadcastingFieldOpMap,
#   b::AbstractMatrix,
#   a::AbstractVector)

#   @check size(a,1) == size(b,1)
#   np, ni = size(b)
#   setsize!(cache,(np,ni))
#   r = cache.array
#   for p in 1:np
#     ap = a[p]
#     for i in 1:ni
#       r[p,i] = f.op(b[p,i],ap)
#     end
#   end
#   r
# end

op2 = v⋅f
ax = map(i->i(x),op2.args)
# lazy_map(Fields.BroadcastingFieldOpMap(op2.op.op),ax...)
item = ax[1][1],ax[2][1]
k = Fields.BroadcastingFieldOpMap(op2.op.op)
cache = return_cache(k,item...)
evaluate!(cache,k,item...)

grid = TensorProductGrid(domain,partition)
tpgrid = CartesianGrid(domain,partition)

@assert all(get_node_coordinates(grid) .== get_node_coordinates(tpgrid))
