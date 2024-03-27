using Gridap
using Gridap.FESpaces
using ForwardDiff
using BlockArrays
using LinearAlgebra
using Test
using Gridap.Algebra
using Gridap.ODEs
using Gridap.Helpers
using Gridap.Fields
using Gridap.MultiField
using BlockArrays
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 1
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

n = 10
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(x,μ,t) = 1+exp(-sin(2π*t/tf)^2*(1-x[2])/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

inflow(μ,t) = 1-cos(2π*t/tf)+sin(μ[2]*2π*t/tf)/μ[1]
g(x,μ,t) = VectorValue(-x[2]*(1-x[2])*inflow(μ,t),0.0)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

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
induced_norm((du,dp),(v,q)) = ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
trial_u = TransientTrialParamFESpace(test_u,gμt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,coupling,trian_res,trian_jac,trian_jac_t)
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,induced_norm,ptspace,
  trial,test,trian_res,trian_jac)
feop = TransientParamLinNonlinFEOperator(feop_lin,feop_nlin)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
fesolver = ThetaMethod(nls,dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ,RB.SpaceOnlyMDEIM();nsnaps_state=50,nsnaps_test=1,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("navier_stokes","toy_mesh")))

fesnaps,festats = ode_solutions(rbsolver,feop,xh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

# fesnaps = Serialization.deserialize(RB.get_snapshots_filename(test_dir))

println(RB.space_time_error(results))
save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

# POD-MDEIM error
pod_err,mdeim_error = RB.pod_mdeim_error(rbsolver,feop,rbop,fesnaps)

son = select_snapshots(fesnaps,RB.online_params(rbsolver))
r = get_realization(son)
trial = get_trial(rbop)(r)
fe_trial = get_fe_trial(rbop)(r)
x̂ = zero_free_values(trial)
y = zero_free_values(fe_trial)

odecache = allocate_odecache(fesolver,rbop,r,(y,))
# solve!((x̂,),fesolver,rbop,r,(y,),odecache)

odecache_lin,odecache_nlin = odecache
odeslvrcache_nlin,odeopcache_nlin = odecache_nlin
A_lin,b_lin = jacobian_and_residual(fesolver,get_linear_operator(rbop),r,(y,),odecache_lin)
A_nlin = allocate_jacobian(get_nonlinear_operator(rbop),r,(y,),odeopcache_nlin)
b_nlin = allocate_residual(get_nonlinear_operator(rbop),r,(y,),odeopcache_nlin)
sysslvrcache = ((A_lin,A_nlin),(b_lin,b_nlin))
sysslvr = fesolver.sysslvr
stageop = get_stage_operator(fesolver,rbop,r,(y,),odecache_nlin)
solve!(x̂,y,sysslvr,stageop,sysslvrcache)

odeop = stageop.odeop
red_trial = get_trial(odeop)(stageop.rx)
A_cache,b_cache = sysslvrcache
B = residual!(b_cache,stageop,y)
A = jacobian!(A_cache,stageop,y)
B .+= A*x̂

dx̂ = similar(B)
ss = symbolic_setup(nls.ls,A)
ns = numerical_setup(ss,A)

# v = get_fe_basis(test_u)
# du = get_trial_fe_basis(test_u)
# u = rand(num_free_dofs(test_u))
# _trial_u = TrialFESpace(test_u,x->x)
# uh = FEFunction(_trial_u,u)
# β = assemble_vector(v->c(uh,v,dΩ),test_u)

# test_c(u,du,v) = ∫(v⊙(∇(du)'⋅u))dΩ

# γ = assemble_vector(v->test_c(uh,zero(_trial_u),v),test_u)
# ε = assemble_matrix((u,v)->test_c(uh,u,v),_trial_u,test_u)*u

# β ≈ γ+ε

s1 = select_snapshots(fesnaps,1)
intp_err = RB.interpolation_error(rbsolver,feop,rbop,s1)
# proj_err = RB.linear_combination_error(rbsolver,feop,rbop,s1)
err_lin = RB.linear_combination_error(rbsolver,feop.op_linear,rbop.op_linear,s1;name="linear")
# err_nlin = RB.linear_combination_error(rbsolver,feop.op_nonlinear,rbop.op_nonlinear,s1;name="non linear")
odeop = get_algebraic_operator(feop.op_nonlinear)
# errA,errb = linear_combination_error(rbsolver,odeop,rbop.op_nonlinear,s1)
feA,feb = jacobian_and_residual(fesolver,odeop,s1)
feA_comp = RB.compress(fesolver,feA,get_trial(rbop.op_nonlinear),get_test(rbop.op_nonlinear))
feb_comp = RB.compress(fesolver,feb,get_test(rbop.op_nonlinear))
# rbA,rbb = jacobian_and_residual(rbsolver,rbop.op_nonlinear,s1)

x = get_values(s1)
r = get_realization(s1)

odecache = allocate_odecache(fesolver,rbop.op_nonlinear,r,(x,))
# jacobian_and_residual(fesolver,rbop.op_nonlinear,r,(x,),odecache)
stageop = get_stage_operator(fesolver,rbop.op_nonlinear,r,(x,),odecache)
A = jacobian(stageop,x)

# lin
_odecache = allocate_odecache(fesolver,rbop.op_linear,r,(x,))
# jacobian_and_residual(fesolver,rbop.op_linear,r,(x,),odecache)
_stageop = get_stage_operator(fesolver,rbop.op_linear,r,(x,),_odecache)
_A = jacobian(_stageop,x)

# AA = allocate_jacobian(stageop,x)
# jacobian!(AA,stageop,x)
# A
odeop,odeopcache = stageop.odeop,stageop.odeopcache
rx = stageop.rx
usx = stageop.usx(x)
allocate_jacobian(odeop,rx,usx,odeopcache)
