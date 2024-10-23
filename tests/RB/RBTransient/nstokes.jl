using Gridap
using Gridap.Algebra
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

using ReducedOrderModels.FEM
using ReducedOrderModels.TProduct
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamFESpaces
using ReducedOrderModels.ParamSteady
using ReducedOrderModels.ParamODEs

using ReducedOrderModels.RB
using ReducedOrderModels.RB.RBSteady
using ReducedOrderModels.RB.RBTransient

θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 10*dt

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1,0,1)
partition = (3,3,3)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"no_slip",collect(1:22))
add_tag_from_tags!(labels,"no_penetration",[23,24])
add_tag_from_tags!(labels,"inlet",[25])
add_tag_from_tags!(labels,"outlet",[26])

order = 2
degree = 2*order+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1]*exp((sin(t)+cos(t))/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 0.5
inflow(μ,t) = abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100)
g_in(x,μ,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(x,μ,t) = VectorValue(0.0,0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

res_nlin(μ,t,(u,p),(v,q),dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["no_slip","inlet","no_penetration"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_0,gμt_in,gμt_in])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_jac,trian_jac_t)
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,ptspace,
  trial,test,trian_res,trian_jac)
feop = LinearNonlinearTransientParamFEOperator(feop_lin,feop_nlin)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

fesolver = ThetaMethod(NewtonRaphsonSolver(LUSolver(),1e-10,20),dt,θ)

tol = 1e-4
state_reduction = TransientReduction(coupling,tol,energy;nparams=50,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=20,nparams_jac=20,nparams_djac=1)

fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon,xh0μ)
perf = rb_performance(rbsolver,rbop,x,x̂,festats,rbstats,μon)

################################################################################
