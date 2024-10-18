using Gridap
using Test
using DrWatson

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)
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

h(x,μ) = abs(cos(μ[3]))
h(μ) = x->h(x,μ)
hμ(μ) = ParamFunction(h,μ)

g(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)

stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = ParamTrialFESpace(test,gμ)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,trian_res,trian_stiffness)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
state_reduction = Reduction(tol,energy;nparams=100,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

test_dir = datadir("poisson")
create_dir(test_dir)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
save(test_dir,rbop)

μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,rbop,x,x̂,festats,rbstats,μon)
