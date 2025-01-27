module SteadyStokesPOD

using DrWatson
using Gridap
using ROM

import Gridap.MultiField: BlockMultiFieldStyle

include("ExamplesInterface.jl")

pranges = (1,10,-1,5,1,2)
pspace = ParamSpace(pranges)

model_dir = datadir(joinpath("models","model_circle_h007.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"no_slip",["cylinders","walls"])
add_tag_from_tags!(labels,"no_penetration",["top_bottom"])
add_tag_from_tags!(labels,"dirichlet",["inlet"])

order = 2
degree = 2*(order)+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(μ) = x -> μ[1]*exp(-μ[2]*x[1])
aμ(μ) = ParamFunction(a,μ)

g_in(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0,0.0)
gμ_in(μ) = ParamFunction(g_in,μ)
g_0(μ) = x -> VectorValue(0.0,0.0,0.0)
gμ_0(μ) = ParamFunction(g_0,μ)

a(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
l(μ,(u,p),(v,q),dΩ) = a(μ,(u,p),(v,q),dΩ)

trian_l = (Ω,)
trian_a = (Ω,)
domains = FEDomains(trian_l,trian_a)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["no_slip","no_penetration","inlet"],
  dirichlet_masks=[(true,true,true),(false,false,true),(true,true,true)])
trial_u = ParamTrialFESpace(test_u,[gμ_0,gμ_in,gμ_in])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = 1e-5
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=80,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=40,nparams_jac=40)

dir = datadir("stokes_pod")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
ExamplesInterface.run_test(dir,rbsolver,feop,tols)

end
