using Gridap
using Test
using DrWatson
using Serialization

using GridapEmbedded

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

R  = 0.5
L = 0.8*(2*R)
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(L,0.0)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R
dp = pmax - pmin

n = 30
partition = (n,n)
bgmodel = TProductModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo3)
Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩ_out = Measure(Ω_out,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)

nΓ = get_normal_vector(Γ)
nΓg = get_normal_vector(Γg)

const γd = 10.0
const γg = 0.1
const h = dp[1]/n
const ν0 = 1e-3

ν(x,μ) = 1+exp(-x[1]/sum(μ))
ν(μ) = x->ν(x,μ)
νμ(μ) = ParamFunction(ν,μ)

f(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g(x,μ) = 0.0
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)

# a(μ,u,v,dΩ,dΩ_out,dΓ,dΓg) = ( ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( ∇(v)⋅∇(u) )dΩ_out
#   + ∫( (γd/h)*v*u  - νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
#   + ∫( (γg*h)*jump(nΓg⋅∇(v))*jump(nΓg⋅∇(u)) ) * dΓg
#   )

# b(μ,u,v,dΩ,dΓ) = ∫( (γd/h)*v*fμ(μ) - νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( (γd/h)*v*u - (nΓ⋅∇(v))*u ) * dΓ

# non-symmetric formulation for stability

a(μ,u,v,dΩ,dΩ_out,dΓ,dΓg) = ( ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( ν0*∇(v)⋅∇(u) )dΩ_out
  + ∫( (-1)*νμ(μ)*v*(nΓ⋅∇(u)) + νμ(μ)*(nΓ⋅∇(v))*u )dΓ
  + ∫( (γg*h)*jump(nΓg⋅∇(v))*jump(nΓg⋅∇(u)) ) * dΓg
  )

b(μ,u,v,dΩ,dΓ) = ∫( v*fμ(μ) - νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( (γd/h)*v*gμ(μ) - (nΓ⋅∇(v))*gμ(μ) ) * dΓ

reffe = ReferenceFE(lagrangian,Float64,order)

domains = FEDomains((Ω,Γ),(Ω,Ω_out,Γ,Γg))
test = TProductFESpace(Ωbg,reffe,conformity=:H1;dirichlet_tags=["boundary"])
trial = ParamTrialFESpace(test,gμ)
feop = LinearParamFEOperator(b,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
state_reduction = TTSVDReduction(tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10,random=true)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon,Ω)
