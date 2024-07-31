using Gridap
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using LinearAlgebra
using Test
using DrWatson
using Serialization

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

using Mabla.FEM
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

# # THIS WORKS
# # time marching
# θ = 0.5
# dt = 0.01
# t0 = 0.0
# tf = 0.1

# # parametric space
# pranges = fill([1,10],3)
# tdomain = t0:dt:tf
# ptspace = TransientParamSpace(pranges,tdomain)

# # geometry
# n = 10
# domain = (0,1,0,1)
# partition = (n,n)
# model = CartesianDiscreteModel(domain, partition)
# labels = get_face_labeling(model)
# add_tag_from_tags!(labels,"neumann",[8])
# add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])

# order = 1
# degree = 2*order
# Ω = Triangulation(model)
# dΩ = Measure(Ω,degree)
# Γn = BoundaryTriangulation(model,tags=["neumann"])
# dΓn = Measure(Γn,degree)

# # weak formulation
# a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
# a(μ,t) = x->a(x,μ,t)
# aμt(μ,t) = TransientParamFunction(a,μ,t)

# f(x,μ,t) = 1.
# f(μ,t) = x->f(x,μ,t)
# fμt(μ,t) = TransientParamFunction(f,μ,t)

# h(x,μ,t) = abs(cos(t/μ[3]))
# h(μ,t) = x->h(x,μ,t)
# hμt(μ,t) = TransientParamFunction(h,μ,t)

# g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
# g(μ,t) = x->g(x,μ,t)
# gμt(μ,t) = TransientParamFunction(g,μ,t)

# u0(x,μ) = 0
# u0(μ) = x->u0(x,μ)
# u0μ(μ) = ParamFunction(u0,μ)

# stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
# mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
# rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
# res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

# trian_res = (Ω,Γn)
# trian_stiffness = (Ω,)
# trian_mass = (Ω,)

# induced_norm(du,v) = ∫(du*v)dΩ + ∫(∇(v)⋅∇(du))dΩ

# reffe = ReferenceFE(lagrangian,Float64,order)
# test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
# trial = TransientTrialParamFESpace(test,gμt)
# feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
#   trial,test,trian_res,trian_stiffness,trian_mass)
# uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# P = JacobiLinearSolver()
# fesolver = LinearSolvers.FGMRESSolver(10,P;restart=true,rtol=1.e-8,verbose=true)#LinearSolvers.CGSolver(P;rtol=1.e-8,verbose=true)
# odesolver = ThetaMethod(fesolver,dt,θ)

# r = realization(feop;nparams=5)

# ϵ = 1e-5
# rbsolver = RBSolver(odesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
# fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ;r)

# fesolver′ = LUSolver()
# odesolver′ = ThetaMethod(fesolver′,dt,θ)
# rbsolver′ = RBSolver(odesolver′,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
# fesnaps′,festats′ = fe_solutions(rbsolver′,feop,uh0μ;r)

# r = realization(feop)
# r0 = ParamDataStructures.get_at_time(r,:initial)
# r = r0
# odeop = get_algebraic_operator(feop.op)
# us0 = (get_free_dof_values(uh0μ(r0.params)),)
# odecache = allocate_odecache(odesolver,odeop,r0,us0)
# state0,cache = ode_start(odesolver,odeop,r0,us0,odecache)
# statef = copy.(state0)

# odeslvrcache,odeopcache = odecache
# reuse,A,b,sysslvrcache = odeslvrcache

# sysslvr = odesolver.sysslvr
# x = statef[1]
# fill!(x,zero(eltype(x)))
# dtθ = θ*dt
# shift!(r,dtθ)
# usx = (state0[1],x)
# ws = (dtθ,1)
# update_odeopcache!(odeopcache,odeop,r)
# stageop = LinearParamStageOperator(odeop,odeopcache,r,usx,ws,A,b,reuse,sysslvrcache)

# sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)


# ############################## GRIDAP
# μ = get_params(r0).params[1]
# t = 0.005

# _g(x) = g(x,μ,t)
# _trial = TrialFESpace(test,_g)
# _uh = zero(_trial)
# _x = get_free_dof_values(_uh)
# _A = param_getindex(stageop.A,1)
# _b = collect(param_getindex(stageop.b,1))
# _ss = symbolic_setup(fesolver,_A)
# _ns = numerical_setup(_ss,_A)
# solve!(_x,_ns,_b)

# function Gridap.Algebra.numerical_setup!(ns::LinearSolvers.JacobiNumericalSetup,A::AbstractMatrix)
#   println(typeof(A))
#   ns.inv_diag .= 1.0 ./ diag(a)
# end
# #################################################################
# using Gridap.MultiField

# θ = 1.0
# dt = 0.01
# t0 = 0.0
# tf = 0.1

# pranges = fill([1,10],3)
# tdomain = t0:dt:tf
# ptspace = TransientParamSpace(pranges,tdomain)

# n = 10
# domain = (0,1,0,1)
# partition = (n,n)
# model = CartesianDiscreteModel(domain, partition)
# labels = get_face_labeling(model)
# add_tag_from_tags!(labels,"inlet",[7])
# add_tag_from_tags!(labels,"dirichlet0",[1,2,3,4,5,6])

# order = 2
# degree = 2*order
# Ω = Triangulation(model)
# dΩ = Measure(Ω,degree)

# a(x,μ,t) = μ[1]*exp(sin(π*t/tf)*x[1]/sum(μ))
# a(μ,t) = x->a(x,μ,t)
# aμt(μ,t) = TransientParamFunction(a,μ,t)

# inflow(μ,t) = 1-cos(π*t/tf)+sin(π*t/(μ[2]*tf))/μ[3]
# g_in(x,μ,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
# g_in(μ,t) = x->g_in(x,μ,t)
# gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
# g_w(x,μ,t) = VectorValue(0.0,0.0)
# g_w(μ,t) = x->g_w(x,μ,t)
# gμt_w(μ,t) = TransientParamFunction(g_w,μ,t)
# g_c(x,μ,t) = VectorValue(0.0,0.0)
# g_c(μ,t) = x->g_c(x,μ,t)
# gμt_c(μ,t) = TransientParamFunction(g_c,μ,t)

# u0(x,μ) = VectorValue(0.0,0.0)
# u0(μ) = x->u0(x,μ)
# u0μ(μ) = ParamFunction(u0,μ)
# p0(x,μ) = 0.0
# p0(μ) = x->p0(x,μ)
# p0μ(μ) = ParamFunction(p0,μ)

# α = 1.e6
# Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,order-1;quad_order=degree)
# graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
# stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ + graddiv(u,v,dΩ)
# mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
# res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

# trian_res = (Ω,)
# trian_stiffness = (Ω,)
# trian_mass = (Ω,)

# coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
# induced_norm((du,dp),(v,q)) = (∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ)*(1/dt)

# xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

# reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
# test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","dirichlet0"])
# trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w])
# reffe_p = ReferenceFE(lagrangian,Float64,order-1)
# test_p = TestFESpace(model,reffe_p;conformity=:C0)
# trial_p = TrialFESpace(test_p)
# test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
# trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
# feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
#   trial,test,coupling,trian_res,trian_stiffness,trian_mass)

# solver_u = LUSolver()
# solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)

# diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,test_p,test_p)]
# bblocks = map(CartesianIndices((2,2))) do I
#   (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
# end
# coeffs = [1.0 1.0;
#           0.0 1.0]
# P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
# solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6,verbose=true)
# odesolver = ThetaMethod(solver,dt,θ)

# r = realization(feop;nparams=5)

# ϵ = 1e-5
# rbsolver = RBSolver(odesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
# fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ;r)

# fesolver′ = LUSolver()
# odesolver′ = ThetaMethod(fesolver′,dt,θ)
# rbsolver′ = RBSolver(odesolver′,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
# fesnaps′,festats′ = fe_solutions(rbsolver′,feop,xh0μ;r)

# r = realization(feop)
# r0 = ParamDataStructures.get_at_time(r,:initial)
# r = r0
# odeop = get_algebraic_operator(feop.op)
# us0 = (get_free_dof_values(xh0μ(r0.params)),)
# odecache = allocate_odecache(odesolver,odeop,r0,us0)
# state0,cache = ode_start(odesolver,odeop,r0,us0,odecache)
# statef = copy.(state0)

# odeslvrcache,odeopcache = odecache
# reuse,A,b,sysslvrcache = odeslvrcache

# sysslvr = odesolver.sysslvr
# x = statef[1]
# fill!(x,zero(eltype(x)))
# dtθ = θ*dt
# shift!(r,dtθ)
# usx = (state0[1],x)
# ws = (dtθ,1)
# update_odeopcache!(odeopcache,odeop,r)
# stageop = LinearParamStageOperator(odeop,odeopcache,r,usx,ws,A,b,reuse,sysslvrcache)

# # sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)

# ss = symbolic_setup(sysslvr,stageop.A)
# ns = numerical_setup(ss,stageop.A)
# rmul!(stageop.b,-1)

# solve!(x,ns,b)

# μ = get_params(r0).params[1]
# t = 0.005

# _g_in(x) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
# _g_w(x) = VectorValue(0.0,0.0)
# _trial_u = TrialFESpace(test_u,[_g_in,_g_w])
# _test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
# _trial = MultiFieldFESpace([_trial_u,trial_p];style=BlockMultiFieldStyle())

# _uh = zero(_trial)
# _x = get_free_dof_values(_uh)
# _A = param_getindex(stageop.A,1)
# _b = param_getindex(stageop.b,1)
# _ss = symbolic_setup(fesolver,_A)
# _ns = numerical_setup(_ss,_A)
# solve!(_x,_ns,_b)

# make it work

θ = 1.0
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:2dt
ptspace = TransientParamSpace(pranges,tdomain)

n = 5
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"inlet",[7])
add_tag_from_tags!(labels,"dirichlet0",[1,2,3,4,5,6])

order = 2
degree = 2*(order+1)
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1]
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

inflow(μ,t) = -μ[2]*t
g_in(x,μ,t) = VectorValue(inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_w(x,μ,t) = VectorValue(0.0,0.0)
g_w(μ,t) = x->g_w(x,μ,t)
gμt_w(μ,t) = TransientParamFunction(g_w,μ,t)

α = 1.e2
Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,order-1;quad_order=degree,space=:P)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ + graddiv(u,v,dΩ)
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = (∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ)*(1/dt)

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_w])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) #conformity=:C0)#
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)

diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,test_p,test_p)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]
solver_u = LUSolver()
# solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)
solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=true)
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6,verbose=true)
odesolver = ThetaMethod(solver,dt,θ)

r = realization(feop;nparams=2)

ϵ = 1e-5
rbsolver = RBSolver(odesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ;r)

# fesolver′ = LUSolver()
# odesolver′ = ThetaMethod(fesolver′,dt,θ)
# rbsolver′ = RBSolver(odesolver′,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
# fesnaps′,festats′ = fe_solutions(rbsolver′,feop,xh0μ;r)

#
using Gridap.ODEs
μ = get_params(r).params[2]

_a(x,t) = a(x,μ,t)
_a(t) = x->_a(x,t)
_stiffness(t,(u,p),(v,q),dΩ) = ∫(_a(t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ + graddiv(u,v,dΩ)
_stiffness(t,u,v) = _stiffness(t,u,v,dΩ)
_mass(t,(u,p),(v,q),dΩ) = ∫(v⋅u)dΩ
_mass(t,u,v) = _mass(t,u,v,dΩ)
_g_in(x,t) = g_in(x,μ,t)
_g_in(t) = x->_g_in(x,t)
_g_w(x,t) = g_w(x,μ,t)
_g_w(t) = x->_g_w(x,t)
_rhs(t,(v,q),dΩ) = ∫(VectorValue(0.0,0.0)⋅v)dΩ
_rhs(t,v) = _rhs(t,v,dΩ)

_trial_u = TransientTrialFESpace(test_u,[_g_in,_g_w])
_trial = TransientMultiFieldParamFESpace([_trial_u,trial_p];style=BlockMultiFieldStyle())
_feop = TransientLinearFEOperator((_stiffness,_mass),_rhs,_trial,test)

_xh0 = interpolate_everywhere([u0μ(get_params(r)[1]),p0μ(get_params(r)[1])],_trial(0.0))

fesltn = solve(odesolver,_feop,t0,2dt,_xh0)
UU,PP = [],[]
for (t_n,xhs_n) in fesltn
  uhs_n,phs_n = xhs_n
  push!(UU,copy(get_free_dof_values(uhs_n)))
  push!(PP,copy(get_free_dof_values(phs_n)))
end

# COMPARE GRIDAP WITH MINE
# MINE
using Gridap.Algebra

r0 = ParamDataStructures.get_at_time(r,:initial)
odeop = get_algebraic_operator(feop.op)
us0 = (get_free_dof_values(xh0μ(r0.params)),)
odecache = allocate_odecache(odesolver,odeop,r0,us0)
state0,cache = ode_start(odesolver,odeop,r0,us0,odecache)
statef = copy.(state0)

odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache

sysslvr = odesolver.sysslvr
x = statef[1]
fill!(x,zero(eltype(x)))
dtθ = θ*dt
shift!(r0,dtθ)
usx = (state0[1],x)
ws = (dtθ,1)
update_odeopcache!(odeopcache,odeop,r0)
stageop = LinearParamStageOperator(odeop,odeopcache,r0,usx,ws,A,b,reuse,sysslvrcache)
# # B = residual!(b,odeop,r0,usx,odeopcache)
# uh = ODEs._make_uh_from_us(odeop,usx,odeopcache.Us)
# v = get_fe_basis(test)
# assem = get_param_assembler(odeop.op,r0)
# fill!(b,zero(eltype(b)))
# dc = get_res(odeop.op)(get_params(r0),get_times(r0),uh,v)
# dcΩ = dc[Ω]

# # r1 = (∫(aμt(get_params(r0),get_times(r0))*∇(v[1])⊙∇(uh[1]))dΩ)[Ω]
# # r1[1][1]

# # r1 = (∫(aμt(get_params(r0),get_times(r0))*∇(v[1])⊙∇(uh[1]))dΩ - ∫(uh[2]*(∇⋅(v[1])))dΩ + ∫(v[2]*(∇⋅(uh[1])))dΩ)[Ω]
# # r1[1][1]

# #wrong
# # r2 = graddiv(uh[1],v[1],dΩ)[Ω]
# # r2[1][1]

# using Gridap.ReferenceFEs
# using Gridap.CellData
# using Gridap.Arrays

# dv = get_data(Π_Qh(divergence(v[1])))
# dv[1][1]

# duh = get_data(Π_Qh(divergence(uh[1])))
# duh[1][1]

# v1 = v[1]
# z1 = uh[1]

# x = get_cell_points(Ω)

# dv1 = divergence(v1)
# dz1 = divergence(z1)
# pdz1 = Π_Qh(dz1)
# pdz1x = pdz1(x)

# cache = return_cache(get_data(pdz1)[1],get_data(x)[1])
# evaluate!(cache,get_data(pdz1)[1],get_data(x)[1])

# d1 = get_data(pdz1)[1]
# d12 = d1.data[2]

# _uh = ODEs._make_uh_from_us(_odeop,_usx,_odeopcache.Us)
# _z1 = _uh[1]
# _dz1 = divergence(_z1)
# _pdz1 = Π_Qh(_dz1)
# _pdz1x = _pdz1(x)

# _cache = return_cache(get_data(_pdz1)[1],get_data(x)[1])

# # evaluate!(nothing,Π_Qh,dz1)
# f_data = CellData.get_data(dz1)
# # fk_data = lazy_map(k,f_data)

# # f_cache = return_cache(Π_Qh,f_data[1])
# k = Π_Qh
# q = get_shapefuns(k.reffe)
# pq = get_coordinates(k.quad)
# wq = get_weights(k.quad)
# lq = ParamDataStructures.BroadcastOpParamFieldArray(⋅,q,f_data[1])
# eval_cache = return_cache(lq,pq)
# lqx = evaluate!(eval_cache,lq,pq)
# integration_cache = return_cache(Fields.IntegrationMap(),lqx,wq)

# using Gridap.Fields
# lq2 = Fields.BroadcastOpFieldArray(⋅,q,param_getindex(f_data[1],2))
# eval_cache2 = return_cache(lq2,pq)
# lqx2 = evaluate!(eval_cache2,lq2,pq)

# # evaluate!(f_cache,Π_Qh,f_data[1])
# eval_cache,integration_cache = cache
# q = get_shapefuns(k.reffe)
# lq = ParamDataStructures.BroadcastOpParamFieldArray(⋅,q,f)
# lqx = evaluate!(eval_cache,lq,get_coordinates(k.quad))
# bq = evaluate!(integration_cache,Fields.IntegrationMap(),lqx,get_weights(k.quad))
# λ = ldiv!(k.Mq,bq)

# GRIDAP
_odeop = get_algebraic_operator(_feop)
_us0 = (get_free_dof_values(_xh0),)
_odecache = allocate_odecache(odesolver,_odeop,t0,_us0)
_state0,_cache = ode_start(odesolver,_odeop,t0,_us0,_odecache)
_statef = copy.(_state0)

_odeslvrcache,_odeopcache = _odecache
_reuse,_A,_b,_sysslvrcache = _odeslvrcache

_x = _statef[1]
fill!(_x,zero(eltype(_x)))
_usx = (_state0[1],_x)
t = t0+θ*dt
update_odeopcache!(_odeopcache,_odeop,t)
_stageop = LinearStageOperator(_odeop,_odeopcache,t,_usx,ws,_A,_b,_reuse,_sysslvrcache)
# # _uh = ODEs._make_uh_from_us(_odeop,_usx,_odeopcache.Us)
# _assem = get_assembler(_odeop.tfeop)
# fill!(_b,zero(eltype(_b)))
# _dc = get_res(_odeop.tfeop)(t,_uh,v)
# forms = get_forms(_odeop.tfeop)
# ∂tkuh = _uh
# for k in 0:1
#   form = forms[k+1]
#   _dc = _dc + form(t, ∂tkuh, v)
#   if k < 1
#     ∂tkuh = ∂t(∂tkuh)
#   end
# end
# # _sysslvrcache = solve!(_x,sysslvr,_stageop,_sysslvrcache)
# _dcΩ = _dc[Ω]
