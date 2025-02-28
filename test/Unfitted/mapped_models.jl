using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using ROManifolds
using ROManifolds.ParamDataStructures

domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)

φ(x) = VectorValue(x[2],3*x[1])
φt(x) = VectorValue(3*x[2],x[1])
mmodel = MappedDiscreteModel(model,φ)

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model,tags=8)
Ωm = Triangulation(mmodel)
Γm = BoundaryTriangulation(mmodel,tags=8)

dΩ = Measure(Ω,4)
dΓ = Measure(Γ,4)
dΩm = Measure(Ωm,4)
dΓm = Measure(Γm,4)

g(x) = x[1]+x[2]

reffe = ReferenceFE(lagrangian,Float64,2)
V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
U = TrialFESpace(V,g)
Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
Um = TrialFESpace(Vm,g)

ν(x) = exp(-x[1])
f(x) = x[2]

atrian(u,v,dΩ) = ∫(ν*∇(v)⋅∇(u))dΩ
btrian(v,dΩ,dΓ) = ∫(f*v)dΩ + ∫(f*v)dΓ

a(u,v) = atrian(u,v,dΩ)
b(v) = btrian(v,dΩ,dΓ)
am(u,v) = atrian(u,v,dΩm)
bm(v) = btrian(v,dΩm,dΓm)

op = AffineFEOperator(a,b,U,V)
opm = AffineFEOperator(am,bm,Um,Vm)

uh = solve(op)
uhm = solve(opm)

v = get_fe_basis(V)
u = get_trial_fe_basis(V)
jcell = a(u,v)[Ω]

vm = get_fe_basis(Vm)
um = get_trial_fe_basis(Vm)
jcellm = am(um,vm)[Ωm]

detJφ = 3
Jφt = CellField(∇(φ),Ω)
νm = ν∘φ
mappedj = (∫( νm*(inv(Jφt)⋅∇(v)) ⋅ (inv(Jφt)⋅∇(u))*detJφ )dΩ)[Ω]

ncells = num_cells(Ω)
compare = lazy_map(≈,jcellm,mappedj)
@assert sum(compare) == ncells


#


μ = Realization([[3.0],[4.0]])
ϕ(μ) = x->VectorValue(x[2],μ[1]*x[1])
ϕμ(μ) = parameterize(ϕ,μ)
mmodel = MappedDiscreteModel(model,ϕμ(μ))

Ωm = Triangulation(mmodel)
Γm = BoundaryTriangulation(mmodel,tags=8)

glue = get_glue(Γm,Val(1))
@test glue.tface_to_mface === Γm.glue.face_to_bgface
glue = get_glue(Γm,Val(2))
@test glue.tface_to_mface === Γm.glue.face_to_cell
face_s_q = glue.tface_to_mface_map

s1 = Point(0.0)
s2 = Point(0.5)
s = [s1,s2]
face_to_s = Fill(s,length(face_s_q))

face_to_q = lazy_map(evaluate,face_s_q,face_to_s)
@test isa(face_to_q,Geometry.FaceCompressedVector)

cell_grid = get_grid(get_background_model(Γm))
cell_shapefuns = get_cell_shapefuns(cell_grid)
cell_grad_shapefuns = lazy_map(Broadcasting(∇),cell_shapefuns)

face_shapefuns = lazy_map(Reindex(cell_shapefuns),glue.tface_to_mface)
face_grad_shapefuns = lazy_map(Reindex(cell_grad_shapefuns),glue.tface_to_mface)

face_shapefuns_q = lazy_map(evaluate,face_shapefuns,face_to_q)
test_array(face_shapefuns_q,collect(face_shapefuns_q))
@test isa(face_shapefuns_q,Geometry.FaceCompressedVector)

face_grad_shapefuns_q = lazy_map(evaluate,face_grad_shapefuns,face_to_q)
test_array(face_grad_shapefuns_q,collect(face_grad_shapefuns_q))
@test isa(face_grad_shapefuns_q,Geometry.FaceCompressedVector)

face_to_nvec = get_facet_normal(Γm)
face_to_nvec_s = lazy_map(evaluate,face_to_nvec,face_to_s)
test_array(face_to_nvec_s,collect(face_to_nvec_s))

glue = Γm.glue
cell_grid = get_grid(get_background_model(Γm.trian))

## Reference normal
function f(r)
  p = get_polytope(r)
  lface_to_n = get_facet_normal(p)
  lface_to_pindex_to_perm = get_face_vertex_permutations(p,num_cell_dims(p)-1)
  nlfaces = length(lface_to_n)
  lface_pindex_to_n = [ fill(lface_to_n[lface],length(lface_to_pindex_to_perm[lface])) for lface in 1:nlfaces ]
  lface_pindex_to_n
end
ctype_lface_pindex_to_nref = map(f, get_reffes(cell_grid))
face_to_nref = Geometry.FaceCompressedVector(ctype_lface_pindex_to_nref,glue)
face_s_nref = lazy_map(constant_field,face_to_nref)

# Inverse of the Jacobian transpose
cell_q_x = get_cell_map(cell_grid)
cell_q_Jt = lazy_map(∇,cell_q_x)
cell_q_invJt = lazy_map(Operation(pinvJt),cell_q_Jt)

d = 1
node_coords = get_node_coordinates(mmodel)
cell_node_ids = Table(get_face_nodes(mmodel,d))
cell_ctype = collect1d(get_face_type(mmodel,d))
ctype_reffe = get_reffaces(ReferenceFE{d},mmodel)

cell_coords = lazy_map(Broadcasting(Reindex(node_coords)),cell_node_ids)
ctype_shapefuns = map(get_shapefuns,ctype_reffe)
cell_shapefuns = expand_cell_data(ctype_shapefuns,cell_ctype)
default_cell_map = lazy_map(linear_combination,cell_coords,cell_shapefuns)
ctype_poly = map(get_polytope,ctype_reffe)
ctype_q0 = map(p->zero(first(get_vertex_coordinates(p))),ctype_poly)
cell_q0 = expand_cell_data(ctype_q0,cell_ctype)
default_cell_grad = lazy_map(∇,default_cell_map)
origins = lazy_map(evaluate,default_cell_map,cell_q0)
# i_to_values = default_cell_map.args[1]
# i_to_basis = default_cell_map.args[2]
# i_to_basis_x = lazy_map(evaluate,i_to_basis,cell_q0)
# # lazy_map(Fields.LinearCombinationMap(:),i_to_values,i_to_basis_x)
# k = Fields.LinearCombinationMap(:)
# cache = return_cache(k,i_to_values[1],i_to_basis_x[1])
# # @which evaluate!(cache,Fields.LinearCombinationMap(1),i_to_values[1],i_to_basis_x[1])
# evaluate!(cache,k,i_to_values[1],i_to_basis_x[1])
gradients = lazy_map(evaluate,default_cell_grad,cell_q0)
cell_map = lazy_map(Fields.affine_map,gradients,origins)

h = cell_map[1]
x1 = Point(0,)
x2 = Point(1,)
x3 = Point(2,)
x = [x1,x2,x3]
h(x)
∇(h)(x)

origin = Point(1,1)
g1 = TensorValue(2,0,0,2)
h1 = AffineField(g1,origin)
hx = h1(x)

CIAO
# node_coords = get_node_coordinates(model)
# cell_node_ids = Table(get_face_nodes(model,d))
# cell_ctype = collect1d(get_face_type(model,d))
# ctype_reffe = get_reffaces(ReferenceFE{d},model)
# cell_coords = lazy_map(Broadcasting(Reindex(node_coords)),cell_node_ids)
# ctype_shapefuns = map(get_shapefuns,ctype_reffe)
# cell_shapefuns = expand_cell_data(ctype_shapefuns,cell_ctype)
# default_cell_map = lazy_map(linear_combination,cell_coords,cell_shapefuns)
# ctype_poly = map(get_polytope,ctype_reffe)
# ctype_q0 = map(p->zero(first(get_vertex_coordinates(p))),ctype_poly)
# cell_q0 = expand_cell_data(ctype_q0,cell_ctype)
# default_cell_grad = lazy_map(∇,default_cell_map)
# # origins = lazy_map(evaluate,default_cell_map,cell_q0)
# i_to_values = default_cell_map.args[1]
# i_to_basis = default_cell_map.args[2]
# i_to_basis_x = lazy_map(evaluate,i_to_basis,cell_q0)
# # lazy_map(Fields.LinearCombinationMap(:),i_to_values,i_to_basis_x)
# k = Fields.LinearCombinationMap(:)
# cache = return_cache(k,i_to_values[1],i_to_basis_x[1])
# evaluate!(cache,k,i_to_values[1],i_to_basis_x[1])
# # gradients = lazy_map(evaluate,default_cell_grad,cell_q0)
# # cell_map = lazy_map(Fields.affine_map,gradients,origins)
