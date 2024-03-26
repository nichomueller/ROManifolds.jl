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
tf = 0.05

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
feop = TransientParamLinearNonlinearFEOperator(feop_lin,feop_nlin)

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

son = select_snapshots(fesnaps,51)
ron = get_realization(son)
θ == 0.0 ? dtθ = dt : dtθ = dt*θ

r = copy(ron)
FEM.shift!(r,dt*(θ-1))

rb_trial = get_trial(rbop)(r)
fe_trial = get_fe_trial(rbop)(r)
red_x = zero_free_values(rb_trial)
y = zero_free_values(fe_trial)
z = similar(y)
z .= 0.0

ode_cache = allocate_cache(rbop,r)
cache_lin = ODETools._allocate_matrix_and_vector(rbop.op_linear,r,y,ode_cache)
cache_nlin = ODETools._allocate_matrix_and_vector(rbop.op_nonlinear,r,y,ode_cache)
cache = cache_lin,cache_nlin

ode_cache = update_cache!(ode_cache,rbop,r)
nlop = RBThetaMethodParamOperator(rbop,r,dtθ,y,ode_cache,z)
# solve!(red_x,fesolver.nls,nlop,cache)

# # fex = copy(nlop.u0)
# fex = copy(get_values(son))
# (cache_jac_lin,cache_res_lin),(cache_jac_nlin,cache_res_nlin) = cache

# # linear res/jac, now they are treated as cache
# lop = nlop.odeop.op_linear
# A_lin,b_lin = ODETools._matrix_and_vector!(cache_jac_lin,cache_res_lin,lop,r,dtθ,y,ode_cache,z)
# cache_jac = A_lin,cache_jac_nlin
# cache_res = b_lin,cache_res_nlin
# cache = cache_jac,cache_res

# # initial nonlinear res/jac
# b = residual!(cache_res,nlop,fex)
# b1 = copy(b)
# A = jacobian!(cache_jac,nlop,fex)
# A1 = copy(A)
# dx = similar(b)
# ss = symbolic_setup(LUSolver(),A)
# ns = numerical_setup(ss,A)

# trial = get_trial(nlop.odeop)(nlop.r)
# isconv, conv0 = Algebra._check_convergence(nls,b)

# rmul!(b,-1)
# solve!(dx,ns,b)
# red_x .+= dx

# xr = recast(red_x,trial)
# fex = xr
# # fex .= recast(red_x,trial)

# b = residual!(cache_res,nlop,fex)
# isconv = Algebra._check_convergence(nls,b,conv0)
# println(maximum(abs,b))

# A = jacobian!(cache_jac,nlop,fex)
# numerical_setup!(ns,A)

# reduced c, x = 0.0
x = similar(get_values(son))
x .= 0.0
A_nlin,b_nlin = ODETools._allocate_matrix_and_vector(rbop.op_nonlinear,r,y,ode_cache)
b_nlin = residual!(b_nlin,rbop.op_nonlinear,r,(x,y),ode_cache)
b_nlin_snap = Snapshots(recast(b_nlin,trial),r)

# full order c, x = 0.0
Us,Uts,fecache = ode_cache
xh = EvaluationFunction(Us[1],x)
bok = allocate_residual(feop.op_nonlinear,r,xh,fecache)

Xh, = ode_cache
dxh = ()
xh = TransientCellField(EvaluationFunction(Xh[1],x),dxh)
residual!(bok,feop.op_nonlinear,r,xh,ode_cache)
bok_snap = Snapshots(bok[1],r)

feA,feb = RB._jacobian_and_residual(fesolver,feop.op_nonlinear,son)
feA_comp = RB.compress(rbsolver,feA,get_trial(rbop),get_test(rbop))
feb_comp = RB.compress(rbsolver,feb,get_test(rbop))
rbA,rbb = RB._jacobian_and_residual(rbsolver,rbop.op_nonlinear,son)
errA = RB._rel_norm(feA_comp,rbA)
errb = RB._rel_norm(feb_comp,rbb)
