using Gridap
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

using ReducedOrderModels

θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 60*dt

pranges = fill([1.0,10.0],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

model_dir = datadir(joinpath("models","new_model_circle_2d.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet0",["walls_p","walls","cylinders_p","cylinders"])
add_tag_from_tags!(labels,"dirichlet",["inlet"])

order = 2
degree = 2*order+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

const Re = 100.0
a(x,μ,t) = μ[1]/Re
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

const W = 0.5
inflow(μ,t) = abs(1-cos(2π*t/tf)+μ[3]*sin(μ[2]*2π*t/tf)/100)
g_in(x,μ,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(x,μ,t) = VectorValue(0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)
domains_lin = FEDomains(trian_res,(trian_jac,trian_jac_t))
domains_nlin = FEDomains(trian_res,(trian_jac,))

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet","dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,domains_lin;constant_forms=(false,true))
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,ptspace,
  trial,test,domains_nlin)
feop = LinearNonlinearTransientParamFEOperator(feop_lin,feop_nlin)

fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

tol = 1e-4
state_reduction = TransientReduction(coupling,tol,energy;nparams=50)#,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=40,nparams_djac=1)

test_dir = datadir(joinpath("navier-stokes","model_circle_2d"))
create_dir(test_dir)

# ################## solve steady stokes problem for IC ##########################

# steady_trial_u = ParamTrialFESpace(test_u,[μ -> gμt_in(μ,dt),μ -> gμt_0(μ,dt)])
# steady_test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
# steady_trial = MultiFieldParamFESpace([steady_trial_u,trial_p];style=BlockMultiFieldStyle())

# steady_a(μ,(u,p),(v,q)) = ∫(aμt(μ,dt)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
# steady_res(μ,(u,p),(v,q)) = (-1)*steady_a(μ,(u,p),(v,q))

# steady_feop_lin = LinearParamFEOperator(steady_a,steady_res,ptspace.parametric_space,steady_trial,steady_test)
# steady_fesolver = LinearFESolver(LUSolver())
# steady_rbsolver = RBSolver(steady_fesolver,state_reduction)

# r = realization(steady_feop_lin;nparams=50)
# uh,stats = solve(steady_fesolver,steady_feop_lin,r)
# vals = get_free_dof_values(uh)
# rt = TransientRealization(r,tdomain)

# ronline = realization(steady_feop_lin;nparams=5)
# uhonline,statsonline = solve(steady_fesolver,steady_feop_lin,ronline)
# valsonline = get_free_dof_values(uhonline)
# rtonline = TransientRealization(ronline,tdomain)

# ################################################################################

fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)
save(test_dir,fesnaps)
rbop = reduced_operator(rbsolver,feop,fesnaps)
save(test_dir,rbop)
r = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,r)

x,festats = solution_snapshots(rbsolver,feop,r,xh0μ)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,r)
println(perf)

# # plotting

# r = get_realization(fesnaps)
# S′ = flatten_snapshots(fesnaps)
# S1 = S′[1][:,:,1]
# r1 = r[1,:]
# U1 = trial_u(r1)
# plt_dir = datadir("plts")
# create_dir(plt_dir)
# for i in 1:length(r1)
#   Ui = param_getindex(U1,i)
#   uhi = FEFunction(Ui,S1[:,i])
#   writevtk(Ω,joinpath(plt_dir,"u_$i.vtu"),cellfields=["uh"=>uhi])
# end
# S2 = S′[2][:,:,1]
# for i in 1:length(r1)
#   Pi = trial_p
#   phi = FEFunction(Pi,S2[:,i])
#   writevtk(Ω,joinpath(plt_dir,"p_$i.vtu"),cellfields=["ph"=>phi])
# end

# STOKES
tol = 1e-4
state_reduction = TransientReduction(coupling,tol,energy;nparams=50,sketch=:sprn)
fesolver_lin = ThetaMethod(LUSolver(),dt,θ)
rbsolver_lin = RBSolver(fesolver_lin,state_reduction;nparams_res=50,nparams_jac=20,nparams_djac=1)
fesnaps,festats = solution_snapshots(rbsolver_lin,feop_lin,xh0μ)
rbop = reduced_operator(rbsolver_lin,feop_lin,fesnaps)
r = realization(feop_lin;nparams=10)
x̂,rbstats = solve(rbsolver_lin,rbop,r)

x,festats = solution_snapshots(rbsolver_lin,feop_lin,r,xh0μ)
perf = rb_performance(rbsolver_lin,feop_lin,rbop,x,x̂,festats,rbstats,r)
println(perf)
