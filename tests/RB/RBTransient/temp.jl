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
using Mabla.RB.RBTransient

θ = 1#0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 10
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet0",[1,2,3,4,5,6])
add_tag_from_tags!(labels,"dirichlet",[7])
add_tag_from_tags!(labels,"neumann",[8])

order = 2
degree = 2*(order)+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1]*exp((sin(t)+cos(t))/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 1
inflow(μ,t) = abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100)
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

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_0,gμt_in])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)

fesolver = ThetaMethod(LUSolver(),dt,θ)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

supr_operator((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

tol = 1e-4
state_reduction = TransientReduction(supr_operator,tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=20,nparams_jac=20,nparams_djac=20)

# fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# ronline = realization(feop;nparams=10)
# x̂,rbstats = solve(rbsolver,rbop,ronline)

# x,festats = solution_snapshots(rbsolver,feop,ronline,xh0μ)
# perf = rb_performance(rbsolver,rbop,x,x̂,festats,rbstats,ronline)

################################################################################

using Gridap.Algebra

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,ptspace,
  trial,test,trian_res,trian_stiffness)
feop = LinNonlinTransientParamFEOperator(feop_lin,feop_nlin)

nlsolver = NewtonRaphsonSolver(LUSolver(),1e-5,40)
fesolver = ThetaMethod(nlsolver,dt,θ)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=20,nparams_jac=20,nparams_djac=20)

fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
ronline = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,ronline;verbose=true)

x,festats = solution_snapshots(rbsolver,feop,ronline,xh0μ)
perf = rb_performance(rbsolver,rbop,x,x̂,festats,rbstats,ronline)

using Gridap.FESpaces
using Gridap.ODEs

op = rbop
solver = rbsolver
r = ronline

# RBSteady.init_online_cache!(solver,op,r)
y = zero_free_values(get_fe_trial(op)(r))
x̂ = zero_free_values(get_trial(op)(r))

odecache = allocate_odecache(fesolver,op,r,(y,))
rbcache = RBSteady.allocate_rbcache(op,r)

nls = fesolver.sysslvr

stageop = get_stage_operator(fesolver,op,r,(y,),odecache,rbcache)
x = y
# solve!(x̂,nls,stageop,r,x)

Â_lin = stageop.lop.A
syscache,U = stageop.cache
Â_cache,b̂_cache = syscache

dx̂ = similar(x̂)
Â = jacobian!(Â_cache,stageop,x)
b̂ = residual!(b̂_cache,stageop,x)
b̂ .+= Â_lin*x̂

inv_project!(x,U,x̂)

Âlin = stageop.lop.A
(Alin,_),_... = stageop.linear_caches

odeop,odeopcache = stageop.odeop,stageop.odeopcache
rx = stageop.rx
usx = stageop.usx(x)
ws = stageop.ws
# Ânlin = jacobian!((Alin,Â_nlin),odeop,rx,usx,ws,odeopcache)
# fe_jacobian!(Alin,odeop,rx,usx,ws,odeopcache)

# b̂ = residual!(b̂_nlin,stageop,x)
# @. b̂ = b̂ + Â_lin*x̂

# ss = symbolic_setup(nls.ls,Â)
# ns = numerical_setup(ss,Â)

# max0 = maximum(abs,b̂)
# tol = 1e-6*max0

# k = 1
# rmul!(b̂,-1)
# solve!(dx̂,ns,b̂)
# @. x̂ = x̂ + dx̂
# inv_project!(x,U,x̂)

# b̂ = residual!(b̂_nlin,stageop,x)
# Â = jacobian!(Â_nlin,stageop,x)
# numerical_setup!(ns,Â)

# @. b̂ = b̂ + Â_lin*x̂
