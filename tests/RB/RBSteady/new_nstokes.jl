using Gridap
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

using ROM

# parametric space
pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

order = 2
degree = 2*order+1

const Re′ = 100.0
a(x,μ) = μ[1]/Re′
a(μ) = x->a(x,μ)
aμ(μ) = ParamFunction(a,μ)

const W = 0.5
inflow(μ) = abs(sin(μ[2]*μ[3]))
g_in(x,μ) = VectorValue(-x[2]*(W-x[2])*inflow(μ),0.0,0.0)
g_in(μ) = x->g_in(x,μ)
gμ_in(μ) = ParamFunction(g_in,μ)
g_0(x,μ) = VectorValue(0.0,0.0,0.0)
g_0(μ) = x->g_0(x,μ)
gμ_0(μ) = ParamFunction(g_0,μ)

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
domains_lin = FEDomains(trian_res,trian_jac)
domains_nlin = FEDomains(trian_res,trian_jac)

c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

jac_lin(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
res_lin(μ,(u,p),(v,q),dΩ) = jac_lin(μ,(u,p),(v,q),dΩ)

res_nlin(μ,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,
  dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
  dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
trial_u = ParamTrialFESpace(test_u,[gμ_in,gμ_in,gμ_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

feop_lin = LinearParamFEOperator(res_lin,jac_lin,pspace,trial,test,domains_lin)
feop_nlin = ParamFEOperator(res_nlin,jac_nlin,pspace,trial,test,domains_nlin)
feop = LinearNonlinearParamFEOperator(feop_lin,feop_nlin)

test_dir = datadir(joinpath("steady-navier-stokes","model_circle_$h"))
create_dir(test_dir)

fesolver = NonlinearFESolver(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true))

tol = 1e-4
state_reduction = Reduction(coupling,tol,energy;nparams=100,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=50)

fesnaps,festats = solution_snapshots(rbsolver,feop)
save(test_dir,fesnaps)
println(festats)
# fesnaps = load_snapshots(test_dir)
rbop = reduced_operator(rbsolver,feop,fesnaps)
save(test_dir,rbop)
ronline = realization(feop;nparams=10,sampling=:uniform)
xonline,festats = solution_snapshots(rbsolver,feop;r=ronline)
save(test_dir,xonline;label="online")
# xonline = load_snapshots(test_dir;label="online")
# ronline = get_realization(xonline)
x̂,rbstats = solve(rbsolver,rbop,ronline)
println(rbstats)
perf = eval_performance(rbsolver,feop,rbop,xonline,x̂,festats,rbstats,ronline)
println(perf)
