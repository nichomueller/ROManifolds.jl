using Gridap
using Gridap.Algebra
using Gridap.MultiField
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

θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 0.15

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

model_dir = datadir(joinpath("models","new_model_circle_2d.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet_noslip",["walls","cylinders","walls_p","cylinders_p"])

order = 2
degree = 2*order+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = μ[1]*exp((sin(t)+cos(t))/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 0.5
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

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
induced_norm((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","dirichlet_noslip"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
# test_p = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_jac,trian_jac_t)
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,induced_norm,ptspace,
  trial,test,trian_res,trian_jac)
feop = LinNonlinTransientParamFEOperator(feop_lin,feop_nlin)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0*p*q)dΩ,test_p,test_p)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]
solver_u = LUSolver()
solver_p = LUSolver()
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-7,verbose=true)
nlsolver = NewtonRaphsonSolver(solver,1e-10,20)
odesolver = ThetaMethod(solver,dt,θ)
lu_nlsolver = NewtonRaphsonSolver(LUSolver(),1e-10,20)
lu_odesolver = ThetaMethod(lu_nlsolver,dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(odesolver,ϵ;nparams_state=50,nparams_res=50,nparams_jac=30,nparams_test=10)
lu_rbsolver = RBSolver(lu_odesolver,ϵ;nparams_state=50,nparams_res=50,nparams_jac=30,nparams_test=10)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("navier_stokes","model_circle_2d")))

fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)
save(test_dir,fesnaps)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats,cache = solve(lu_rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

# save(test_dir,rbop)
save(test_dir,results)

show(results.timer)
println(compute_speedup(results))

# using Gridap.FESpaces
# using Gridap.ODEs

# op = rbop # op = get_algebraic_operator(feop)
# s = select_snapshots(fesnaps,RBSteady.online_params(rbsolver))
# r = get_realization(s)
# fesolver = get_fe_solver(lu_rbsolver)
# red_trial = get_trial(op)(r)
# fe_trial = get_fe_trial(op)(r)
# x̂ = zero_free_values(red_trial)
# y = zero_free_values(fe_trial)
# odecache = allocate_odecache(fesolver,op,r,(y,))

# x = y
# odecache_lin,odecache_nlin = odecache
# odeslvrcache_nlin,odeopcache_nlin = odecache_nlin

# lop = get_linear_operator(op)
# nlop = get_nonlinear_operator(op)

# A_lin,b_lin = jacobian_and_residual(fesolver,lop,r,(y,),odecache_lin;update_cache=false)
# A_nlin = allocate_jacobian(nlop,r,(y,),odeopcache_nlin)
# b_nlin = allocate_residual(nlop,r,(y,),odeopcache_nlin)
# sysslvrcache = ((A_lin,A_nlin),(b_lin,b_nlin))
# sysslvr = fesolver.sysslvr

# using BlockArrays
# B = blocks(A_lin[1][1])

# stageop = get_stage_operator(fesolver,op,r,(y,),odecache_nlin;update_cache=false)
# solve!(x̂,x,sysslvr,stageop,sysslvrcache)
# A_cache,b_cache = sysslvrcache
# # A = jacobian!(A_cache,stageop,x)
# odeop,odeopcache = stageop.odeop,stageop.odeopcache
# r = stageop.rx
# us = stageop.usx(x)
# ws = stageop.ws
# # jacobian!(A_cache,odeop,rx,usx,ws,odeopcache)
# Â_lin,A_nlin = A_cache
# # fe_sA_nlin =
# nop = odeop.op_nonlinear
# red_cache,red_r,red_times,red_us,red_odeopcache = RBTransient._select_fe_quantities_at_time_locations(
#   A_nlin,nop.lhs,r,us,odeopcache)
# # AA = jacobian!(red_cache,nop.op,red_r,red_us,ws,red_odeopcache)
# jacobian!(red_cache,nop.op.op,r,us,ws,odeopcache)
# i = get_matrix_index_map(nop.op)
# sA = Snapshots(red_cache,i,r)
