using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

const L = 1
const R  = 0.1
const n = 30

const domain = (0,L,0,L)
const partition = (n,n)

p1 = Point(0.3,0.5)
p2 = Point(0.7,0.5)
geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = !union(geo1,geo2)

bgmodel = TProductModel(domain,partition)
cutgeo = cut(bgmodel,geo3)

labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"wall",collect(1:6))
add_tag_from_tags!(labels,"Γn",[7,8])

Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γd = EmbeddedBoundary(cutgeo)
Γn = BoundaryTriangulation(bgmodel;tags="Γn")

n_Γd = get_normal_vector(Γd)
n_Γn = get_normal_vector(Γn)

order = 2
degree = 2*order

dΩ = Measure(Ω,degree)
dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)
dΓd = Measure(Γd,degree)
dΓn = Measure(Γn,degree)

ν(x,μ) = μ[1]*exp(-μ[2])
ν(μ) = x->ν(x,μ)
νμ(μ) = ParamFunction(ν,μ)

u(x,μ) = VectorValue(x[1]*x[1],x[2])*abs(μ[1]*sin(μ[2]))
u(μ) = x->u(x,μ)
uμ(μ) = ParamFunction(u,μ)

p(x,μ) = (x[1]-x[2])*abs(cos(μ[3]))
p(μ) = x->p(x,μ)
pμ(μ) = ParamFunction(p,μ)

f(x,μ) = - Δ(u(μ))(x) + ∇(p(μ))(x)
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g(x,μ) = (∇⋅u(μ))(x)
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)

g_0(x,μ) = VectorValue(0.0,0.0)
g_0(μ) = x->g_0(x,μ)
gμ_0(μ) = ParamFunction(g_0,μ)

const ν0 = 1e-3
const γ = order*(order+1)
const h = L/n

a(μ,(u,p),(v,q),dΩ_in,dΩ_out,dΓd) =
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p )dΩ_in +
  ∫( ν0*(∇(v)⊙∇(u) + q*p) )dΩ_out +
  ∫( (γ/h)*v⋅u - v⋅(n_Γd⋅∇(u)) - (n_Γd⋅∇(v))⋅u + (p*n_Γd)⋅v + (q*n_Γd)⋅u )dΓd

l(μ,(u,p),(v,q),dΩ_in,dΓd,dΓn) =
  ∫( v⋅fμ(μ) - q*gμ(μ) )dΩ_in +
  ∫( (γ/h)*v⋅uμ(μ) - (n_Γd⋅∇(v))⋅uμ(μ) + (q*n_Γd)⋅uμ(μ) )dΓd +
  ∫( v⋅(n_Γn⋅∇(uμ(μ))) - (n_Γn⋅v)*pμ(μ) )dΓn -
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p )dΩ_in

trian_res = (Ω_in,Γd,Γn)
trian_jac = (Ω_in,Ω_out,Γd)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags="wall")
trial_u = ParamTrialFESpace(test_u,gμ_0)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)
test_p = TestFESpace(Ω,reffe_p;conformity=:L2)
trial_p = TrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test,trian_res,trian_jac)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=50,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=30,nparams_jac=30)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

r = realization(feop)
fesnaps,festats = solution_snapshots(rbsolver,feop,r)

# MORE INDEX MAPS :(
get_sparse_dof_map(test_u,test_u,Ω_in)
get_sparse_dof_map(test_u,test_u,Ω_out)
get_sparse_dof_map(test_u,test_u,Γd)
get_sparse_dof_map(test_u,test_p,Ω_in)

using ReducedOrderModels.TProduct
using ReducedOrderModels.DofMaps
using SparseArrays

sparsity = SparsityPattern(test_u,test_p)
psparsity = order_sparsity(sparsity,test_u,test_p)
# I,J,_ = findnz(psparsity)
# i,j,_ = DofMaps.univariate_findnz(psparsity)
# g2l_sparse = TProduct._global_2_local(psparsity,I,J,i,j)
# pg2l_sparse = TProduct._permute_dof_map(g2l_sparse,test_u,test_p)
# pg2l = to_nz_index(pg2l_sparse,sparsity)
# SparseDofMap(pg2l,pg2l_sparse,psparsity)
dof_map_I = get_dof_map(test_p)
dof_map_J = get_dof_map(test_u)
dof_map_I_1d = get_tp_dof_dof_map(test_p).indices_1d
dof_map_J_1d = get_tp_dof_dof_map(test_u).indices_1d
