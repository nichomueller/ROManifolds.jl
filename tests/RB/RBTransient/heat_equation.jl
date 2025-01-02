using Gridap
using Test
using DrWatson
using Serialization

using ROM

# time marching
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 10*dt

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

# geometry
model_dir = datadir(joinpath("models","elasticity_3cyl2D.json"))
model = DiscreteModelFromFile(model_dir)
order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

# weak formulation
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
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

energy(du,v) = ∫(du*v)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,trial,test,domains)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# solvers
fesolver = ThetaMethod(LUSolver(),dt,θ)

tol = 1e-4
state_reduction = TransientReduction(tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=20,nparams_jac=20,nparams_djac=20)

test_dir = datadir(joinpath("heateq","elasticity_$(tol)"))
create_dir(test_dir)

# RB method

fesnaps,festats = solution_snapshots(rbsolver,feop,uh0μ)
# fesnaps = load_snapshots(test_dir)
rbop = reduced_operator(rbsolver,feop,fesnaps)
ronline = realization(feop;nparams=10,random=true)
x̂,rbstats = solve(rbsolver,rbop,ronline)

x,festats = solution_snapshots(rbsolver,feop,ronline,uh0μ)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,ronline)
