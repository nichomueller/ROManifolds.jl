using Gridap
using Test
using DrWatson

using ROM

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)
model_dir = datadir(joinpath("models","model_circle_short.json"))
model = DiscreteModelFromFile(model_dir)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["outlet"])
dΓn = Measure(Γn,degree)

a(μ) = x -> 1+exp(-x[1]/sum(μ))
aμ(μ) = ParamFunction(a,μ)

f(μ) = x -> 1.
fμ(μ) = ParamFunction(f,μ)

h(μ) = x -> abs(cos(μ[3]))
hμ(μ) = ParamFunction(h,μ)

g(μ) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
gμ(μ) = ParamFunction(g,μ)

stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
domains = FEDomains(trian_res,trian_stiffness)

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["walls_p","walls_c","walls","inlet"])
trial = ParamTrialFESpace(test,gμ)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
state_reduction = Reduction(tol,energy;nparams=100,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

test_dir = datadir("poisson")
create_dir(test_dir)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
save(test_dir,rbop)

μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

#

using ROM
using Gridap
using DrWatson

# geometry
Ω = (0,1,0,1)
parts = (10,10)
Ωₕ = CartesianDiscreteModel(Ω,parts)
τₕ = Triangulation(Ωₕ)

# parametric quantities
D  = ParamSpace((1,5,1,5))
u(μ) = x -> μ[1]*x[1]^2 + μ[2]*x[2]^2
uₚ(μ) = ParamFunction(u,μ)
f(μ) = x -> -Δ(u(x,μ))
fₚ(μ) = ParamFunction(f,μ)

# numerical integration
order = 1
dΩₕ = Measure(τₕ,2order)

# weak form
rhs(μ,v,dΩₕ) = ∫(fₚ(μ)*v)dΩₕ
lhs(μ,u,v,dΩₕ) = ∫(∇(v)⋅∇(u))dΩₕ

# triangulation information
τₕ_rhs = (τₕ,)
τₕ_lhs = (τₕ,)

# FE interpolation
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(Ωₕ,reffe;dirichlet_tags="boundary")
U = ParamTrialFESpace(V,uₚ)
feop = LinearParamFEOperator(rhs,lhs,D,U,V,τₕ_rhs,τₕ_lhs)

# FE solver
slvr = LinearFESolver(LUSolver())

# RB solver
tol = 1e-4
inner_prod(u,v) = ∫(∇(v)⋅∇(u))dΩₕ
red_sol = PODReduction(tol,inner_prod;nparams=100)
red_rhs = MDEIMReduction(tol;nparams=20)
red_lhs = MDEIMReduction(tol;nparams=1)
rbslvr = RBSolver(slvr,red_sol,red_rhs,red_lhs)

dir = datadir("poisson")
create_dir(dir)

try # try loading offline quantities
    rbop = load_operator(dir,feop)
catch # offline phase
    rbop = reduced_operator(rbslvr,feop)
    save(rbop,dir)
end

# online phase
μ = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbslvr,rbop,μ)

# post process
x,stats = solve(rbslvr,feop,μ)
rb_performance(rbslvr,feop,rbop,x,x̂,stats,rbstats,μ)
