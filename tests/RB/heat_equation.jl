using Gridap
using Test
using DrWatson
using Mabla.FEM
using Mabla.RB

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn

trian_rhs = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),rhs,induced_norm,ptspace,
  trial,test,trian_rhs,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver,dir=test_dir)

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(RB.space_time_error(results))
save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

# POD-MDEIM error
pod_err,mdeim_error = RB.pod_mdeim_error(rbsolver,feop,rbop,fesnaps)

ϵ = 1e-4
rbsolver_space = RBSolver(fesolver,ϵ,RB.SpaceOnlyMDEIM();nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir_space = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver_space,dir=test_dir_space)

rbop_space = reduced_operator(rbsolver_space,feop,fesnaps)
rbsnaps_space,rbstats_space = solve(rbsolver_space,rbop,fesnaps)
results_space = rb_results(rbsolver_space,feop,fesnaps,rbsnaps_space,festats,rbstats_space)

println(RB.space_time_error(results_space))
save(test_dir,rbop_space)
save(test_dir,results_space)

using Gridap.FESpaces
red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
odeop = get_algebraic_operator(feop)
pop = PODOperator(odeop,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
contribs_mat,contribs_vec = jacobian_and_residual(rbsolver,pop,smdeim)

using Gridap.ODEs
using Gridap.Algebra
w0 = get_values(smdeim)
r = get_realization(smdeim)
odecache = ODEs.allocate_odecache(fesolver,odeop,r,(w0,))
odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache
x = copy(w0)
fill!(x,zero(eltype(x)))
dtθ = θ*dt
FEM.shift_time!(r,dt*(θ-1))
us = (x,x)
ws = (1,1/dtθ)
# stageop = LinearParamStageOperator(odeop,odeopcache,r,us,ws,A,b,reuse,sysslvrcache)
# residual!(b,odeop,r,us,odeopcache)
uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
v = get_fe_basis(test)
assem = get_assembler(odeop.op,r)

!add && fill!(b,zero(eltype(b)))

μ,t = get_params(r),get_times(r)

# Residual
res = get_res(odeop.op)
dc = res(μ,t,uh,v)

# Forms
order = get_order(odeop)
forms = get_forms(odeop.op)
∂tkuh = uh
for k in 0:order
  form = forms[k+1]
  dc = dc + form(μ,t,∂tkuh,v)
  if k < order
    ∂tkuh = ∂t(∂tkuh)
  end
end
