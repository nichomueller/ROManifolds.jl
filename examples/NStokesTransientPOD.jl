module ElasticitySteady

using ROM
using Gridap
using DrWatson

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

import Gridap.MultiField: BlockMultiFieldStyle

include("ExamplesInterface.jl")

θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 10*dt

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
state_reduction = TransientReduction(coupling,tol,energy;nparams=10,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=5,nparams_jac=5,nparams_djac=1)

dir = datadir("transient_nstokes_pod")
create_dir(dir)

tols = [1e-5]
ExamplesInterface.run_test(dir,rbsolver,feop,tols,xh0μ)

end

using Gridap.CellData
using SparseArrays
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.Algebra
using Gridap.Arrays
using Gridap.ODEs

using ROM.Utils
using ROM.DofMaps
using ROM.RBSteady
using ROM.RBTransient
using ROM.ParamSteady
using ROM.ParamDataStructures

using BlockArrays
using LinearAlgebra

tol = 1e-4
state_reduction = TransientReduction(tol,energy;nparams=10)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=5,nparams_jac=5,nparams_djac=1)

fesnaps = ExamplesInterface.try_loading_fe_snapshots(dir,rbsolver,feop,xh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=1,random=true)

r = μon
op = rbop

x̂ = zero_free_values(get_trial(op)(r))
x = zero_free_values(get_fe_trial(op)(r))
x0 = get_free_dof_values(xh0μ(get_params(r)))
rbcache = allocate_rbcache(fesolver,op,r,x)

sysslvr = fesolver.sysslvr
ŷ = RBParamVector(x̂,x)
uθ = copy(ŷ)
dut = copy(ŷ)
red_trial = get_trial(op)(r)

function us(u::RBParamVector)
  inv_project!(u.fe_data,rbcache.rbcache.trial,u.data)
  # copyto!(uθ,u)
  # shift!(uθ,red_trial,x0,θ,1-θ)
  # copyto!(dut,u)
  # shift!(dut,red_trial,x0,1/dt,-1/dt)
  (u,u)
end

ws = (1,1)
usx = us(ŷ)

Âcache = jacobian(op,r,usx,ws,rbcache)
b̂cache = residual(op,r,usx,rbcache)

Â_item = testitem(Âcache)
x̂_item = testitem(x̂)
dx̂ = allocate_in_domain(Â_item)
fill!(dx̂,zero(eltype(dx̂)))
ss = symbolic_setup(BackslashSolver(),Â_item)
ns = numerical_setup(ss,Â_item,x̂_item)

shift!(r,dt*(θ-1))
nlop = ParamStageOperator(op,rbcache,r,us,ws)
# Algebra._solve_nr!(ŷ,Âcache,b̂cache,dx̂,ns,sysslvr,nlop)

rmul!(b̂cache,-1)
xi = param_getindex(ŷ,1)
Ai = param_getindex(Âcache,1)
bi = param_getindex(b̂cache,1)
numerical_setup!(ns,Ai)
solve!(dx̂,ns,bi)
xi .+= dx̂

# residual!(b̂cache,nlop,ŷ)
# jacobian!(Âcache,nlop,ŷ)

cache = nlop.cache
r = nlop.r
usx = nlop.us(ŷ)
b = b̂cache
# residual!(b,nlop.op,r,usx,cache)
nnlop = get_nonlinear_operator(nlop.op)
b_lin = RBSteady.linear_residual(cache)
rbcache_nlin = cache.rbcache

b_nlin = residual!(b,nnlop,r,usx,rbcache_nlin)
axpy!(1.0,b_lin,b_nlin)
for (A_lin,u) in zip(cache.A.hypred,usx)
  mul!(b_nlin,A_lin,u,true,true)
end

# odeop = get_algebraic_operator(feop.op_nonlinear)
# # ress = residual_snapshots(rbsolver,odeop,fesnaps)
# sres = fesnaps
# us_res = (get_values(sres),)
# us0_res = RBTransient.get_initial_values(sres)
# r_res = get_realization(sres)
# # residual(fesolver,odeop,r_res,us_res,us0_res)
# u = get_values(sres)
# x = copy(u)
# uθ = copy(u)

# shift!(r,dt*(θ-1))
# shift!(uθ,us0_res,θ,1-θ)
# shift!(x,us0_res,1/dt,-1/dt)
# us = (uθ,x)
# b = residual(odeop,r,us)


inv_project!(ŷ.fe_data,rbcache.rbcache.trial,ŷ.data)
copyto!(uθ.fe_data,u.fe_data)
shift!(uθ.fe_data,x0,θ,1-θ)
copyto!(dut.fe_data,u.fe_data)
shift!(dut.fe_data,x0,1/dt,-1/dt)
(uθ,dut)

#

s1 = flatten_snapshots(fesnaps[1])
projection_space = projection(state_reduction.reduction_space,s1,norm_matrix[Block(1,1)])
proj_s1 = project(projection_space,s1)
proj_s2 = change_mode(proj_s1,num_params(s))
projection_time = projection(get_reduction_time(red),proj_s2)
TransientProjection(projection_space,projection_time)
