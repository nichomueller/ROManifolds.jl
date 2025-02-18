using Gridap
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using ROM

θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 10*dt

pdomain = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

n = 10
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[7])
add_tag_from_tags!(labels,"dirichlet0",collect(1:6))

order = 2
degree = 2*(order)+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(μ,t) = x -> μ[1]*exp((sin(t)+cos(t))/sum(μ))
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 1.0
inflow(μ,t) = abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100)
g_in(μ,t) = x -> VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(μ,t) = x -> VectorValue(0.0,0.0)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

u0(μ) = x -> VectorValue(0.0,0.0)
u0μ(μ) = ParamFunction(u0,μ)
p0(μ) = x -> 0.0
p0μ(μ) = ParamFunction(p0,μ)

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet","dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TransientTrialParamFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,domains;constant_forms=(false,true))

fesolver = ThetaMethod(LUSolver(),dt,θ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

tol = 1e-4
state_reduction = TransientReduction(coupling,tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=20,nparams_djac=1)

fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon,xh0μ)

x,festats = solution_snapshots(rbsolver,feop,μon,xh0μ)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)
