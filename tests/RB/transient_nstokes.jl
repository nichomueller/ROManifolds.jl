using Gridap
using Gridap.FESpaces
using GridapGmsh
using ForwardDiff
using BlockArrays
using LinearAlgebra
using Test
using Gridap.Algebra
using Gridap.ODEs
using Gridap.ODEs.TransientFETools
using Gridap.ODEs.ODETools
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

res_lin(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
jac_lin(μ,t,(u,p),(du,dp),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ + ∫(q*(∇⋅(du)))dΩ
jac_t_lin(μ,t,(u,p),(dut,dpt),(v,q),dΩ) = ∫(v⋅dut)dΩ

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
_feop_lin = AffineTransientParamFEOperator(res_lin,jac_lin,jac_t_lin,induced_norm,ptspace,trial,test,coupling)
feop_lin = FEOperatorWithTrian(_feop_lin,trian_res,trian_jac,trian_jac_t)
_feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,induced_norm,ptspace,trial,test)
feop_nlin = FEOperatorWithTrian(_feop_nlin,trian_res,trian_jac)
feop = TransientParamLinearNonlinearFEOperator(feop_lin,feop_nlin)

xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))
nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
fesolver = ThetaMethod(nls,dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ,RB.SpaceTimeMDEIM();nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("navier_stokes","toy_mesh")))

fesnaps,festats = ode_solutions(rbsolver,feop,xh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(feop,rbsolver,fesnaps,rbsnaps,festats,rbstats)

# fesnaps = Serialization.deserialize(RB.get_snapshots_filename(test_dir))

println(RB.space_time_error(results))
save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ,RB.SpaceOnlyMDEIM();nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("navier_stokes","toy_mesh")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver_space,dir=test_dir_space)

rbop_space = reduced_operator(rbsolver_space,feop,fesnaps)
rbsnaps_space,rbstats_space = solve(rbsolver_space,rbop,fesnaps)
results_space = rb_results(feop,rbsolver_space,fesnaps,rbsnaps_space,festats,rbstats_space)

println(RB.space_time_error(results_space))
save(test_dir,rbop_space)
save(test_dir,results_space)

son = select_snapshots(fesnaps,RB.online_params(rbsolver))
ron = get_realization(son)
θ == 0.0 ? dtθ = dt : dtθ = dt*θ

r = copy(ron)
FEM.shift_time!(r,dt*(θ-1))

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
# solve!(red_x,fesolver.nls,nlop,nl_cache)

fex = similar(nlop.u0)
fex .= 0.0
(cache_jac_lin,cache_res_lin),(cache_jac_nlin,cache_res_nlin) = cache

# linear res/jac, now they are treated as cache
lop = nlop.odeop.op_linear
A_lin,b_lin = ODETools._matrix_and_vector!(cache_jac_lin,cache_res_lin,lop,r,dtθ,y,ode_cache,z)
cache_jac = A_lin,cache_jac_nlin
cache_res = b_lin,cache_res_nlin
cache = cache_jac,cache_res

# # initial nonlinear res/jac
# b = residual!(cache_res,nlop,fex)
# A = jacobian!(cache_jac,nlop,fex)
# dx = similar(b)
# ss = symbolic_setup(LUSolver(),A)
# ns = numerical_setup(ss,A)

# opnl = feop.op_nonlinear.op
# odeopopnl = get_algebraic_operator(opnl)
# nopnl = ThetaMethodParamOperator(odeopopnl,r,dtθ,y,ode_cache,z)
# J = jacobian(nopnl,y)

# rbopnl = rbop.op_nonlinear
# LHS1 = rbopnl.lhs[1]
# RHS1 = rbopnl.rhs

# dC = LHS1[1]
# C = RHS1[1]

# dC1 = dC[1]

# jacobian!(cache_jac,nlop,fex)
uF = fex
vθ = nlop.vθ
# ODETools.jacobians!(cache_jac,nlop.odeop,nlop.r,(uF,vθ),(1.0,1/nlop.dtθ),nlop.ode_cache)
A_lin,cache_nl = cache_jac
fecache_nl, = cache_nl
for i = eachindex(fecache_nl)
  LinearAlgebra.fillstored!(fecache_nl[i],zero(eltype(fecache_nl[i])))
end
# A_nlin = ODETools.jacobians!(cache_nl,nlop.odeop.op_nonlinear,nlop.r,(uF,vθ),(1.0,1/nlop.dtθ),ode_cache)
op,r,xhF,γ = nlop.odeop.op_nonlinear,nlop.r,(uF,vθ),(1.0,1/nlop.dtθ)
fe_A,coeff_cache,lincomb_cache = cache_nl
fe_sA = fe_jacobians!(fe_A,op,r,xhF,γ,ode_cache)
