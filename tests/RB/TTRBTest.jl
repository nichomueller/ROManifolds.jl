using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.Helpers
using Gridap.TensorValues
using BlockArrays
using DrWatson
using Kronecker
using Mabla.FEM
using Mabla.RB

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.05

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1)
partition = (5,5)
model = TProductModel(domain,partition)

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

trian_res = (Ω.trian,Γn)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

induced_norm(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=5,nsnaps_test=5,nsnaps_mdeim=2)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","tt_toy_h1")))

fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
# soff = select_snapshots(fesnaps,RB.offline_params(rbsolver))
# norm_matrix = assemble_norm_matrix(feop)
# cores = ttsvd(soff,norm_matrix)
odeop = get_algebraic_operator(feop)
pop = PODOperator(odeop,red_trial,red_test)
smdeim = select_snapshots(fesnaps,RB.mdeim_params(rbsolver))
A,b = jacobian_and_residual(rbsolver,pop,smdeim)

# fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
# rbop = reduced_operator(rbsolver,feop,fesnaps)
# rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
# results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)

# println(RB.space_time_error(results))
# println(RB.speedup(results))

# save(test_dir,fesnaps)
# save(test_dir,rbop)
# save(test_dir,results)

s = smdeim
odeop = pop.odeop
us = (get_values(s),)
r = get_realization(s)
# odecache = allocate_odecache(fesolver,odeop,r,us)
us0N = (us[1],us[1])
odeopcache = allocate_odeopcache(odeop,r,us0N)
# A = allocate_jacobian(odeop,r,us0N,odeopcache)
uh = ODEs._make_uh_from_us(odeop,us0N,odeopcache.Us)
trial = evaluate(get_trial(odeop.op),nothing)
du = get_trial_fe_basis(trial)
test = get_test(odeop.op)
v = get_fe_basis(test)
assem = get_assembler(odeop.op,r)

μ,t = get_params(r),get_times(r)

jacs = get_jacs(odeop.op)
As = ()

k = 1
jac = jacs[k]
trian_jac = odeop.op.trian_jacs[k]
dc = jac(μ,t,uh,du,v)
A = contribution(trian_jac) do trian
  matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
  allocate_matrix(assem,matdata)
end
