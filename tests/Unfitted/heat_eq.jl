using Gridap
using Test
using DrWatson
using Serialization

using GridapEmbedded

using ReducedOrderModels

R  = 0.5
L  = 0.5*R
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(-L,L)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R

n = 30
partition = (n,n)
bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
dp = pmax - pmin

cutgeo = cut(bgmodel,geo3)

Ω_act = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Vstd = TestFESpace(Ω_act,reffe,conformity=:H1)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

V = AgFEMSpace(Vstd,aggregates)
U = TrialFESpace(V)

degree = 2*order+1
dΩ = Measure(Ω,degree)

Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

const γd = 10.0    # Nitsche coefficient
const h = dp[1]/n  # Mesh size according to the parameters of the background grid

# add parameters
pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

f(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

trians = (Ω,Γ)

stiffness(μ,u,v,dΩ,dΓ) = ∫( ∇(v)⋅∇(u) )dΩ + ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
rhs(μ,v,dΩ,dΓ) = ∫( (γd/h)*v*fμ(μ) - (n_Γ⋅∇(v))*fμ(μ) )dΓ
res(μ,u,v,dΩ,dΓ) = rhs(μ,v,dΩ,dΓ) - stiffness(μ,u,v,dΩ,dΓ)

trial = ParamTrialFESpace(V,fμ)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,V,trians,trians)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
state_reduction = Reduction(tol,energy;nparams=100,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)
