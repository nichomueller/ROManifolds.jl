using Gridap
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

using ReducedOrderModels

# time marching
θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 20*dt

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

order = 2
degree = 2*order+1

const Re′ = 100.0
a(x,μ,t) = μ[1]/Re′
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 0.5
inflow(μ,t) = abs(1-cos(π*t/tf)+μ[3]*sin(μ[2]*π*t/tf)/100)
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

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

h = "h007"
model_dir = datadir(joinpath("models","model_circle_$h.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
add_tag_from_tags!(labels,"dirichlet",["inlet"])

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)
domains_lin = FEDomains(trian_res,(trian_jac,trian_jac_t))
domains_nlin = FEDomains(trian_res,(trian_jac,))

c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
  dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_0,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1)
trial_p = TransientTrialParamFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,domains_lin;constant_forms=(false,true))
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,ptspace,trial,test,domains_nlin)
feop = LinearNonlinearTransientParamFEOperator(feop_lin,feop_nlin)

fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

test_dir = datadir(joinpath("navier-stokes-t20","model_circle_$h"))
create_dir(test_dir)

tol = 1e-4
state_reduction = TransientReduction(coupling,tol,energy;nparams=50,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=20,nparams_jac=20,nparams_djac=1)

fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)
save(test_dir,fesnaps)
# println(festats)
# fesnaps = load_snapshots(test_dir)
rbop = reduced_operator(rbsolver,feop,fesnaps)
# save(test_dir,rbop)
ronline = realization(feop;nparams=10,random=true)
xonline,festats = solution_snapshots(rbsolver,feop,xh0μ;r=ronline)
save(test_dir,xonline;label="online")
# xonline = load_snapshots(test_dir;label="online")
# ronline = get_realization(xonline)
x̂,rbstats = solve(rbsolver,rbop,ronline)
println(rbstats)
perf = rb_performance(rbsolver,feop,rbop,xonline,x̂,festats,rbstats,ronline)
println(perf)

# function param_writevtk(Ω,trial::FESpace,s,dir=datadir("plts");cellfieldnames=String[])
#   create_dir(dir)
#   r = get_realization(s)
#   U = trial(r)
#   sh = FEFunction(U,get_values(s))
#   sh1 = param_getindex(sh,1) # fetch only first parameter
#   celldata = [name => uh for (name,uh) in zip(cellfieldnames,sh1)]
#   param_writevtk(Ω,dir,celldata)
# end

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
#   Pi = trial_p(nothing)
#   phi = FEFunction(Pi,S2[:,i])
#   writevtk(Ω,joinpath(plt_dir,"p_$i.vtu"),cellfields=["ph"=>phi])
# end
