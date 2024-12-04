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
domains = FEDomains(trian_res,trian_jac)

coupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ#∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags="wall")
trial_u = ParamTrialFESpace(test_u,gμ_0)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
trial_p = TrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = fill(1e-4,4)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=10)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=10,nparams_jac=10)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10,random=true)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

# u1 = flatten_snapshots(fesnaps[1])[:,1]
# p1 = flatten_snapshots(fesnaps[2])[:,1]
# r1 = get_realization(fesnaps[1])[1]
# U1 = param_getindex(trial_u(r1),1)
# uh = FEFunction(U1,u1)
# ph = FEFunction(trial_p,p1)
# writevtk(Ω_in,datadir("plts/sol"),cellfields=["uh"=>uh,"ph"=>ph])

# check if supremizers work

supr_form(dp,v) = ∫(dp*(∇⋅(v)))dΩ
BT = assemble_matrix(supr_form,test_p.space,test_u.space)
Φ_u = get_basis(rbop.test.subspace[1])
Φ_p = get_basis(rbop.test.subspace[2])
B̂ = Φ_u'*BT*Φ_p

# standard code
using ReducedOrderModels.RBSteady
using LinearAlgebra
energy_u(du,v) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ
X_primal = assemble_matrix(energy_u,test_u.space,test_u.space)

i = 1
H_primal = cholesky(X_primal)
supr_i = H_primal \ BT * Φ_p
red = state_reduction.reduction
red_style = ReductionStyle(red)
Φ_su = RBSteady.union_bases(PODBasis(Φ_u),supr_i,X_primal,red_style[1]).basis

B̂_ok = Φ_su'*BT*Φ_p
