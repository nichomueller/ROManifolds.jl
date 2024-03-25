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
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_rhs = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_rhs,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=1,nsnaps_mdeim=20)
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

using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ODEs
son = select_snapshots(fesnaps,RB.online_params(rbsolver))
r = get_realization(son)

FEM.shift_time!(r,dt*(θ-1))

red_trial = get_trial(rbop)(r)
fe_trial = get_fe_trial(rbop)(r)
x = zero_free_values(red_trial)
y = zero_free_values(fe_trial)
z = copy(y)
us = (y,z)
ws = (1,1)

odecache = allocate_odecache(fesolver,rbop,r,(y,))
odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache

update_odeopcache!(odeopcache,rbop,r)
stageop = LinearParamStageOperator(rbop,odeopcache,r,us,ws,A,b,reuse,sysslvrcache)
ye=residual!(b,rbop,r,us,odeopcache)
sysslvrcache = solve!(x,fesolver.sysslvr,stageop,sysslvrcache)

s1 = select_snapshots(fesnaps,1)
intp_err = RB.interpolation_error(rbsolver,feop,rbop,s1)
# proj_err = linear_combination_error(solver,feop,rbop,s1)
odeop = get_algebraic_operator(feop)
feA,feb = jacobian_and_residual(fesolver,odeop,s1)
feA_comp = compress(rbsolver,feA,get_trial(rbop),get_test(rbop))
feb_comp = compress(rbsolver,feb,get_test(rbop))
rbA,rbb = jacobian_and_residual(rbsolver,rbop,s1)

feA,feb = FEM.jacobian_and_residual(fesolver,odeop,s1)
rbA,rbb = FEM.jacobian_and_residual(rbsolver,rbop.op,s1)

_odeop = rbop.op.odeop
r = get_realization(s1)
us = (get_values(s1),)
cache = allocate_odecache(fesolver,_odeop,r,us)
w0 = us[1]
odeslvrcache,odeopcache = cache
reuse,A,b,sysslvrcache = odeslvrcache
x = copy(w0)
fill!(x,zero(eltype(x)))
dtθ = θ*dt
FEM.shift_time!(r,dt*(θ-1))
us = (x,x)
ws = (1,1/dtθ)
update_odeopcache!(odeopcache,_odeop,r)
stageop = LinearParamStageOperator(_odeop,odeopcache,r,us,ws,A,b,reuse,sysslvrcache)
FEM.shift_time!(r,dt*(1-θ))
sA = Snapshots(stageop.A,r)
sb = Snapshots(stageop.b,r)

# bb = residual!(b,_odeop,r,us,odeopcache)
uh = ODEs._make_uh_from_us(_odeop,us,odeopcache.Us)
v = get_fe_basis(test)
assem = get_assembler(_odeop.op,r)
fill!(b,zero(eltype(b)))
μ,t = get_params(r),get_times(r)
# Residual
res = get_res(_odeop.op)
dc = res(μ,t,uh,v)
# Forms
order = get_order(_odeop)
forms = get_forms(_odeop.op)
∂tkuh = uh
for k in 0:order
  form = forms[k+1]
  dc = dc + form(μ,t,∂tkuh,v)
  if k < order
    ∂tkuh = ∂t(∂tkuh)
  end
end
trian = _odeop.op.trian_res[1]
dc[trian]
fun(x) = sum(sum(x))
norm(lazy_map(fun,dc[trian][1]))

vecdata = FEM.collect_cell_vector_for_trian(test,dc,trian)
assemble_vector_add!(b.values[1],assem,vecdata)

# ad = rbop.rhs[1]
# ids_space,ids_time = RB.get_indices_space(ad),RB.get_indices_time(ad)
# fes_ids = RB.select_snapshots_entries(reverse_snapshots(feb[1]),ids_space,ids_time)
# rbs_ids = RB.select_snapshots_entries(reverse_snapshots(rbb[1]),ids_space,ids_time)

dtrian = Measure(trian,2)
c1 = -1*∫(fμt(μ,t)*v)dtrian
c2 = stiffness(μ,t,uh,v,dtrian)
c3 = mass(μ,t,∂t(uh),v,dtrian)

norm(fun(c1[trian]))
norm(fun(c2[trian]))
norm(fun(c3[trian]))

norm(fun(c1+c2+c3))
ye = c1+c2+c3

# this is ok
yecdata = FEM.collect_cell_vector_for_trian(test,ye,trian)

dc2 = forms[1](μ,t,uh,v)
dc3 = forms[2](μ,t,∂t(uh),v)
