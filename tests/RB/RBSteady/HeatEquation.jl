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

pranges = fill([1,10],3)
tdomain = t0:dt:tf
pspace = ParamSpace(pranges,tdomain)
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

a(x,μ) = 1+exp(-x[1]/sum(μ))
a(μ) = x->a(x,μ)
aμ(μ) = ParamFunction(a,μ)

f(x,μ) = 1.
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

h(x,μ) = abs(cos(t/μ[3]))
h(μ) = x->h(x,μ)
hμ(μ) = ParamFunction(h,μ)

g(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)

stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)

induced_norm(du,v) = ∫(v*u)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TrialParamFESpace(test,gμ)
feop = ParamLinearFEOperator((stiffness,),res,induced_norm,pspace,
  trial,test,trian_res,trian_stiffness)

ϵ = 1e-4
rbsolver = RBSolver(fesolver,ϵ;nparams_state=50,nparams_test=10,nsnaps_mdeim=20)
test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver,dir=test_dir)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
results = rb_performance(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)

show(results.timer)
save(test_dir,fesnaps)
save(test_dir,rbop)
save(test_dir,results)

# POD-MDEIM error
pod_err,mdeim_error = pod_mdeim_error(rbsolver,feop,rbop,fesnaps)

ϵ = 1e-4
rbsolver_space = RBSolver(fesolver,ϵ;nparams_state=50,nparams_test=10,nsnaps_mdeim=20)
test_dir_space = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","elasticity_h1")))

# we can load & solve directly, if the offline structures have been previously saved to file
# load_solve(rbsolver_space,dir=test_dir_space)

rbop_space = reduced_operator(rbsolver_space,feop,fesnaps)
rbsnaps_space,rbstats_space = solve(rbsolver_space,rbop,fesnaps)
results_space = rb_performance(rbsolver_space,feop,fesnaps,rbsnaps_space,festats,rbstats_space)

println(RB.compute_error(results_space))
save(test_dir,rbop_space)
save(test_dir,results_space)
