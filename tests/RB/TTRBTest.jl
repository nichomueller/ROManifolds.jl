using Gridap
using Test
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (10,10)
model = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

induced_norm(du,v) = ∫(v*u)dΩ + ∫(∇(v)⋅∇(du))dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=5,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ;tt_format=true)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))
println(RB.speedup(results))

save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

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
using Mabla.RB

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)

test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])

reffe1 = ReferenceFE(QUAD,lagrangian,Float64,order)
reffe2 = ReferenceFE(QUAD,lagrangian,VectorValue{2,Float64},order)

_reffe1 = ReferenceFE((SEGMENT,SEGMENT),FEM.tplagrangian,Float64,order)

FEM.get_type(_reffe1)
get_order(_reffe1) == get_order(reffe1)
get_orders(_reffe1) == get_orders(reffe1)
num_dofs(_reffe1) == num_dofs(reffe1)
get_polytope(_reffe1) == get_polytope(reffe1)
pb = get_prebasis(reffe1)
_pb = get_prebasis(_reffe1)

pb.terms == pb.terms && pb.orders == pb.orders

dofs = get_dof_basis(reffe1)
_dofs = get_dof_basis(_reffe1)

CIAO
# struct MyTensorProdStruct{T,D} <: AbstractVector{T}
#   array::Vector{T}
#   MyTensorProdStruct(array::Vector{T}) where T = new{T,length(array)}(array)
# end

# Base.length(v::MyTensorProdStruct) = length(v.array)
# Base.size(v::MyTensorProdStruct) = size(v.array)
# Base.getindex(v::MyTensorProdStruct,i::Int) = getindex(v.array,i)

# abstract type TensorProductRefFE{D} <: ReferenceFE{D} end

# struct LagrangianTensorProdRefFE{C,D} <: TensorProductRefFE{D}
#   reffes::MyTensorProdStruct{GenericLagrangianRefFE{C,1},D}
#   tpreffe::GenericLagrangianRefFE{C,D}
# end

# function TensorProductRefFE(
#   reffes::MyTensorProdStruct{<:GenericLagrangianRefFE},
#   tpreffe::GenericLagrangianRefFE)

#   LagrangianTensorProdRefFE(reffes,tpreffe)
# end

# struct MyTensorProdLagrangian <: ReferenceFEName end

# const tplagrangian = MyTensorProdLagrangian()

# function ReferenceFEs.ReferenceFE(
#   polytope::MyTensorProdStruct{<:Polytope,D},
#   ::MyTensorProdLagrangian,
#   ::Type{T},
#   orders::Union{Integer,Tuple{Vararg{Integer}}};
#   kwargs...) where {T,D}

#   reffes = map(1:D) do i
#     ReferenceFE(polytope[i],lagrangian,T,orders;kwargs...)
#   end |> MyTensorProdStruct
#   tpp = Polytope(ntuple(i->HEX_AXIS,D)...)
#   tproduct_reffe = ReferenceFE(tpp,lagrangian,T,orders;kwargs...)
#   TensorProductRefFE(reffes,tproduct_reffe)
# end

# _basis = tplagrangian
# _ctype_to_polytope = MyTensorProdStruct([SEGMENT,SEGMENT])
# r1 = ReferenceFE(_ctype_to_polytope,_basis,reffe_args...;reffe_kwargs...)

# test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])

# perm1 = RB.get_dof_permutation(model,test,order)

# v = get_fe_basis(test)

# Ω = Triangulation(model)
# dΩ = Measure(Ω,2*order)

# # CartesianDiscreteModel(domain,partition)
# desc = CartesianDescriptor(domain,partition)
# _desc = UnivariateDescriptor(domain,partition)
# _model = UnivariateDiscreteModel(_desc)

T = VectorValue{2,Float64}
p = QUAD
orders = (2,2)

rebasis = compute_monomial_basis(T,p,orders)
nodes, face_own_nodes = compute_nodes(p,orders)
dofs = LagrangianDofBasis(T,nodes)
reffaces = compute_lagrangian_reffaces(T,p,orders)

nnodes = length(dofs.nodes)
ndofs = length(dofs.dof_to_node)
metadata = reffaces
_reffaces = vcat(reffaces...)
face_nodes = ReferenceFEs._generate_face_nodes(nnodes,face_own_nodes,p,_reffaces)
face_own_dofs = ReferenceFEs._generate_face_own_dofs(face_own_nodes, dofs.node_and_comp_to_dof)
face_dofs = ReferenceFEs._generate_face_dofs(ndofs,face_own_dofs,p,_reffaces)

# face_to_num_fnodes = map(num_nodes,_reffaces)
# push!(face_to_num_fnodes,nnodes)

# face_to_lface_to_own_fnodes = map(get_face_own_nodes,_reffaces)
# push!(face_to_lface_to_own_fnodes,face_own_nodes)

# face_to_lface_to_face = get_faces(p)

# result = ReferenceFEs._generate_face_nodes_aux(nnodes,face_own_nodes,face_to_num_fnodes,
#   face_to_lface_to_own_fnodes,face_to_lface_to_face)
