using Gridap
using Test
using DrWatson

using Mabla.FEM
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)
# model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
# model = DiscreteModelFromFile(model_dir)
domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

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

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver,dir=test_dir)

fesnaps,festats = fe_solutions(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

println(RB.compute_error(results))
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

println(RB.compute_error(results_space))
save(test_dir,rbop_space)
save(test_dir,results_space)

solver = rbsolver
fesolver = get_fe_solver(solver)
nparams = num_params(solver)
sol = solve(fesolver,feop,uh0μ;nparams)
odesol = sol.odesol
r = odesol.r
stats = @timed begin
  vals = collect(odesol)
end
i = get_vector_index_map(feop)
snaps = Snapshots(vals,i,r)

using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ODEs

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
op = get_algebraic_operator(feop)
reduced_operator(rbsolver,op,red_trial,red_test,fesnaps)

pop = PODOperator(op,red_trial,red_test)

smdeim = select_snapshots(fesnaps,RBSteady.mdeim_params(rbsolver))
jacs,ress = jacobian_and_residual(rbsolver,pop,smdeim)
combine = (x,y) -> θ*x+(1-θ)*y
# red_jac = reduced_jacobian(rbsolver,pop,jacs)
basis = reduced_basis(jacs[1][1])
lu_interp,integration_domain = mdeim(rbsolver.mdeim_style,basis)

# proj_basis = reduce_operator(rbsolver.mdeim_style,basis,get_basis(red_trial),get_basis(red_test);combine)
b,b_trial,b_test = basis,RBSteady.get_basis(red_trial),RBSteady.get_basis(red_test)

T = Float64
bs = get_basis_space(b)
bt = get_basis_time(b)
bs_trial = get_basis_space(b_trial)
bt_trial = get_basis_time(b_trial)
bs_test = get_basis_space(b_test)
bt_test = get_basis_time(b_test)

T = Float64
s = num_reduced_dofs(b_test),num_reduced_dofs(b),num_reduced_dofs(b_trial)
b̂st = Array{T,3}(undef,s)
b̂t = RBTransient.combine_basis_time(bt,bt_trial,bt_test;combine)

using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ODEs

op = rbop
son = select_snapshots(fesnaps,RBSteady.online_params(rbsolver))
r = get_realization(son)
red_trial = get_trial(op)(r)
fe_trial = get_fe_trial(op)(r)
x̂ = zero_free_values(red_trial)
y = zero_free_values(fe_trial)
odecache = allocate_odecache(fesolver,op,r,(y,))

sysslvr = fesolver.sysslvr
odeslvrcache,odeopcache = odecache
reuse,A,b,sysslvrcache = odeslvrcache

us = (y,y)
ws = (1,1/(dt*θ))
# stageop = get_stage_operator(fesolver,rbop,r,(y,),odecache)
bb = residual!(b,rbop,r,us,odeopcache)
AA = jacobian!(A,rbop,r,us,ws,odeopcache)

# fe_sb = fe_residual!(b,rbop,r,us,odeopcache)
red_cache,red_r,red_times,red_us,red_odeopcache = RBTransient._select_fe_quantities_at_time_locations(
  b,rbop.rhs,r,us,odeopcache)

b = residual!(red_cache,rbop.op,red_r,red_us,red_odeopcache)





# fe_sA = fe_jacobian!(A,rbop,r,us,ws,odeopcache)
red_r,red_times,red_us,red_odeopcache = RBTransient._select_fe_quantities_at_time_locations(rbop.lhs,r,us,odeopcache)
AA = jacobian!(A,op.op,red_r,red_us,ws,red_odeopcache)
