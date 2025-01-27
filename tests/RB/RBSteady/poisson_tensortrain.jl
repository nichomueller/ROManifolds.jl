using Gridap
using Test
using DrWatson
using Serialization

using ROM

# parametric space
pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

# geometry
n = 10
domain = (0,1,0,1)
partition = (n,n)
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])
add_tag_from_tags!(labels,"neumann",[8])

order = 2
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model;tags="neumann")
dΓn = Measure(Γn,degree)

# weak formulation
a(x,μ) = 1+exp(x[1]/sum(μ))
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

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

jac(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
res(μ,u,v,dΩ,dΓn) = jac(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

trian_res = (Ω.trian,Γn)
trian_jac = (Ω.trian,)
domains = FEDomains(trian_res,trian_jac)

energy(du,v) = ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = ParamTrialFESpace(test,gμ)
feop = LinearParamFEOperator(res,jac,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = fill(1e-4,3)
reduction = TTSVDReduction(tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,reduction;nparams_res=20,nparams_jac=20)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
ronline = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,ronline)

x,festats = solution_snapshots(rbsolver,feop,ronline,uh0μ)
perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats,ronline)
