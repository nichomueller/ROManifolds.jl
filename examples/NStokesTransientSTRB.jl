module NStokesTransientSTRB

using ROManifolds
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
tf = 30*dt

pdomain = (1,10,1,10,1,2)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

model_dir = datadir(joinpath("models","back_facing_channel.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)

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

const Lb = 0.2
const Ub = 0.4
inflow(μ,t) = abs(1-cos(2π*t/tf)+sin((2π*t/tf)/μ[2])/μ[2])
g_in(x,μ,t) = VectorValue(μ[3]*(x[2]-Ub)*(x[2]-Lb)*inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(x,μ,t) = VectorValue(0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

u0(μ) = x -> VectorValue(0.0,0.0)
u0μ(μ) = ParamFunction(u0,μ)
p0(μ) = x -> 0.0
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
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","walls"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1)
trial_p = TransientTrialParamFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

feop_lin = TransientParamLinearOperator((stiffness,mass),res,ptspace,
  trial,test,domains_lin;constant_forms=(false,true))
feop_nlin = TransientParamOperator(res_nlin,jac_nlin,ptspace,
  trial,test,domains_nlin)
feop = LinearNonlinearTransientParamOperator(feop_lin,feop_nlin)

fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

tol = 1e-4
state_reduction = TransientReduction(coupling,tol,energy;nparams=60,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=20,nparams_djac=1)

dir = datadir("transient_nstokes_pod")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
ExamplesInterface.run_test(dir,rbsolver,feop,tols,xh0μ;reuse_online=true)

end

# using ROManifolds
# using Gridap
# using DrWatson

# using GridapSolvers
# using GridapSolvers.LinearSolvers
# using GridapSolvers.NonlinearSolvers

# import Gridap.MultiField: BlockMultiFieldStyle

# θ = 1.0
# dt = 0.0025
# t0 = 0.0
# tf = 20*dt

# pdomain = (1,10,1,10,1,2)
# tdomain = t0:dt:tf
# ptspace = TransientParamSpace(pdomain,tdomain)

# domain = (0,1,0,1)
# partition = (10,10)
# model = CartesianDiscreteModel(domain,partition)
# labels = get_face_labeling(model)
# add_tag_from_tags!(labels,"inlet",[7])
# add_tag_from_tags!(labels,"walls",[1,2,3,4,5,6])

# order = 2
# degree = 2*order
# Ω = Triangulation(model)
# dΩ = Measure(Ω,degree)

# const Re = 100.0
# a(x,μ,t) = μ[1]/Re
# a(μ,t) = x->a(x,μ,t)
# aμt(μ,t) = TransientParamFunction(a,μ,t)

# conv(u,∇u) = (∇u')⋅u
# dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
# c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
# dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

# inflow(μ,t) = abs(1-cos(2π*t/tf)+sin((2π*t/tf)/μ[2])/μ[2])
# g_in(x,μ,t) = VectorValue(μ[3]*x[2]*(x[2]-1)*inflow(μ,t),0.0)
# g_in(μ,t) = x->g_in(x,μ,t)
# gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
# g_0(x,μ,t) = VectorValue(0.0,0.0)
# g_0(μ,t) = x->g_0(x,μ,t)
# gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

# u0(μ) = x -> VectorValue(0.0,0.0)
# u0μ(μ) = ParamFunction(u0,μ)
# p0(μ) = x -> 0.0
# p0μ(μ) = ParamFunction(p0,μ)

# stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
# mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
# res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

# res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
# jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

# trian_res = (Ω,)
# trian_jac = (Ω,)
# trian_jac_t = (Ω,)
# domains_lin = FEDomains(trian_res,(trian_jac,trian_jac_t))
# domains_nlin = FEDomains(trian_res,(trian_jac,))

# coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
# energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

# reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
# test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","walls"])
# trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_0])
# reffe_p = ReferenceFE(lagrangian,Float64,order-1)
# test_p = TestFESpace(model,reffe_p;conformity=:H1)
# trial_p = TransientTrialParamFESpace(test_p)
# test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
# trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

# feop_lin = TransientParamLinearOperator((stiffness,mass),res,ptspace,
#   trial,test,domains_lin;constant_forms=(false,true))
# feop_nlin = TransientParamOperator(res_nlin,jac_nlin,ptspace,
#   trial,test,domains_nlin)
# feop = LinearNonlinearTransientParamOperator(feop_lin,feop_nlin)

# fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
# xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

# tol = 1e-4
# state_reduction = TransientReduction(coupling,tol,energy;nparams=60)
# rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=20,nparams_djac=1)


# fesnaps, = solution_snapshots(rbsolver,feop,xh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# μon = realization(feop)
# x̂,rbstats = solve(rbsolver,rbop,μon,xh0μ)

# using ROManifolds.ParamAlgebra
# using ROManifolds.ParamDataStructures
# using ROManifolds.RBTransient
# using ROManifolds.RBSteady
# using ROManifolds.ParamODEs
# using Gridap.FESpaces

# op,r = rbop,μon
# x̂ = zero_free_values(get_trial(op)(r))

# nlop = parameterize(op,r)
# syscache = allocate_systemcache(nlop,x̂)
# ffesolver = RBTransient.get_system_solver(rbsolver)
# # solve!(x̂,ffesolver,nlop,syscache)
# r = ParamODEs._get_realization(nlop)
# ParamODEs.front_shift!(ffesolver,r)
# ParamODEs._update_paramcache!(nlop,r)
# # solve!(x̂,ffesolver.sysslvr,nlop,syscache)
# fill!(x̂,zero(eltype(x̂)))
# ParamAlgebra.update_systemcache!(nlop,x̂)

# A,b = syscache.A,syscache.b
# using Gridap.Algebra
# residual!(b,nlop,x̂)
# jacobian!(A,nlop,x̂)

# # using BlockArrays
# # jfeop = join_operators(set_domains(feop))
# # x1 = mortar(map(ConsecutiveParamArray,get_indexed_data(select_snapshots(fesnaps,1))))
# # ress = residual(jfeop,μon,x1)

# using Gridap.Arrays
# A_item = testitem(A)
# x_item = testitem(x̂)
# dx = allocate_in_domain(A_item)
# fill!(dx,zero(eltype(dx)))
# ss = symbolic_setup(fesolver.sysslvr.ls,A_item)
# ns = numerical_setup(ss,A_item,x_item)

# Algebra._solve_nr!(x,A,b,dx,ns,nls,op)

# ################################################################################

# rbop_old = RBTransient.old_reduced_operator(rbsolver,feop,fesnaps)

# op_old = rbop_old
# xold = zero_free_values(op_old.op_nonlinear.trial.space(r))
# x̂old = zero_free_values(get_trial(op_old)(r)).data
# rbcache = RBTransient.allocate_rbcache(fesolver,op_old,r,xold)
# sysslvr = fesolver.sysslvr

# ŷ = RBParamVector(x̂old,xold)

# function us(u::RBParamVector)
#   inv_project!(u.fe_data,rbcache.rbcache.trial,u.data)
#   (u,u)
# end

# ws = (1,1)
# usx = us(ŷ)
# map(x->fill!(x,0.0),usx)

# Âcache = RBTransient.old_jacobian(op_old,r,usx,ws,rbcache)
# b̂cache = RBTransient.old_residual(op_old,r,usx,rbcache)

# bb = RBTransient.old_allocate_residual(op_old,r,usx,rbcache)
# # old_residual!(bb,op_old,r,usx,rbcache)
# nlop = op_old.op_nonlinear
# A_lin = rbcache.A
# b_lin = rbcache.b
# rbcache_nlin = rbcache.rbcache

# b_nlin = RBTransient.old_residual!(bb,nlop,r,usx,rbcache_nlin)

# Â_item = testitem(Âcache)
# x̂_item = testitem(x̂)
# dx̂ = allocate_in_domain(Â_item)
# fill!(dx̂,zero(eltype(dx̂)))
# ss = symbolic_setup(BackslashSolver(),Â_item)
# ns = numerical_setup(ss,Â_item,x̂_item)

# i = 1
# xi = param_getindex(x,i)
# Ai = param_getindex(A,i)
# bi = param_getindex(b,i)
# numerical_setup!(ns,Ai)
# solve!(dx,ns,bi)
# xi .+= dx

# residual!(b,op,x)
# jacobian!(A,op,x)

# # compare residuals

# # residual!(b,nlop,x̂)
# syscache_lin = get_linear_systemcache(nlop)
# A_lin = get_matrix(syscache_lin)
# b_lin = get_vector(syscache_lin)
# op_nlin = get_nonlinear_operator(nlop)
# residual!(b,op_nlin,x̂)
# mul!(b,A_lin,x̂,1,1)
# axpy!(1,b_lin,b)

# bb = RBTransient.old_allocate_residual(op_old,r,usx,rbcache)
# # RBTransient.old_residual!(bb,op_old,r,usx,rbcache)
# # nlop = get_nonlinear_operator(op)
# A_lin_old = rbcache.A
# b_lin_old = rbcache.b
# rbcache_nlin = rbcache.rbcache
# b_nlin = RBTransient.old_residual!(bb,get_nonlinear_operator(op_old),r,usx,rbcache_nlin)
# axpy!(1.0,b_lin_old,b_nlin)
# mul!(b_nlin,A_lin_old,usx[end],true,true)
