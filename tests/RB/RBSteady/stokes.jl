using Gridap
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using Mabla.FEM
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

domain = (0,1,0,1,0,1)
partition = (3,3,3)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"no_slip",collect(1:22))
add_tag_from_tags!(labels,"np_penetration",[23,24])
add_tag_from_tags!(labels,"inlet",[25])
add_tag_from_tags!(labels,"outlet",[26])

order = 2
degree = 2*(order)+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ) = μ[1]*exp(-μ[2])
a(μ) = x->a(x,μ)
aμ(μ) = ParamFunction(a,μ)

g_in(x,μ) = VectorValue(-x[2]*(1.0-x[2])*abs(μ[3]*sin(μ[2])/10),0.0,0.0)
g_in(μ) = x->g_in(x,μ)
gμ_in(μ) = ParamFunction(g_in,μ)
g_0(x,μ) = VectorValue(0.0,0.0,0.0)
g_0(μ) = x->g_0(x,μ)
gμ_0(μ) = ParamFunction(g_0,μ)

stiffness(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
res(μ,(u,p),(v,q),dΩ) = (-1)*stiffness(μ,(u,p),(v,q),dΩ)

trian_res = (Ω,)
trian_stiffness = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["no_slip","np_penetration","inlet"],
  dirichlet_masks=[(true,true,true),(false,false,true),(true,true,true)])
trial_u = ParamTrialFESpace(test_u,[gμ_0,gμ_in,gμ_in])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,trian_res,trian_stiffness)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=30,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=10,nparams_jac=10)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,rbop,x,x̂,festats,rbstats,μon)

# GRIDAP

μ = get_realization(fesnaps)[1].params
a′(x) = μ[1]*exp(-μ[2])
g_in′(x) = VectorValue(-x[2]*(1.0-x[2])*abs(μ[3]*sin(μ[2])/10),0.0,0.0)
g_0′(x) = VectorValue(0.0,0.0,0.0)

stiffness′((u,p),(v,q)) = ∫(a′*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
res′((v,q)) = ∫(v⋅g_0′)dΩ

U = TrialFESpace(test_u,[g_0′,g_in′,g_in′])
Y = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
X = MultiFieldFESpace([U,trial_p];style=BlockMultiFieldStyle())
feop′ = AffineFEOperator(stiffness′,res′,X,Y)

uh,ph = solve(feop′)
u = get_free_dof_values(uh)
p = get_free_dof_values(ph)
