using Gridap
using Test
using DrWatson
using Serialization

using GridapEmbedded

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

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

order = 1
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

ν(x,μ) = 1+exp(-x[1]/sum(μ))
ν(μ) = x->ν(x,μ)
νμ(μ) = ParamFunction(ν,μ)

f(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g0(x,μ) = 0
g0(μ) = x->g0(x,μ)
g0μ(μ) = ParamFunction(g0,μ)

a(μ,u,v,dΩ,dΩ_out,dΓ,dΓg) = ( ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( ∇(v)⋅∇(u) )dΩ_out
  + ∫( (γd/h)*v*u  - νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
  + ∫( (γg*h)*jump(nΓg⋅∇(v))*jump(nΓg⋅∇(u)) ) * dΓg
  )

b(μ,u,v,dΩ) = ∫( (γd/h)*v*fμ(μ) - νμ(μ)*∇(v)⋅∇(u) )dΩ #+ ∫( (γd/h)*v*u - (nΓ⋅∇(v))*u ) * dΓ

reffe = ReferenceFE(lagrangian,Float64,order)

domains = FEDomains((Ω,),(Ω,Ω_out,Γ,Γg))
test = TProductFESpace(Ωbg,reffe,conformity=:H1;dirichlet_tags=["boundary"])
trial = ParamTrialFESpace(test,g0μ)
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
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

# dof_map_Ω_out = change_domain(_feop.dof_maps.dof_map,Ω_out)
# dof_map_Ω = change_domain(_feop.dof_maps.dof_map,Ω)

# _fesnaps_Ω_out = Snapshots(_fesnaps.data,dof_map_Ω_out,_fesnaps.realization)
# _fesnaps_Ω = Snapshots(_fesnaps.data,dof_map_Ω,_fesnaps.realization)

# _a(μ,u,v,dΩ,dΩ_out,dΓ,dΓg) = ( ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( ∇(v)⋅∇(u) )dΩ_out
#   + ∫( (γd/h)*v*u  - νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
#   + ∫( (γg*h)*jump(nΓg⋅∇(v))*jump(nΓg⋅∇(u)) ) * dΓg
#   )

# b(μ,u,v,dΩ,dΓ) = ∫( (γd/h)*v*fμ(μ) - νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( (γd/h)*v*u - (nΓ⋅∇(v))*u ) * dΓ

# reffe = ReferenceFE(lagrangian,Float64,order)

# _domains = FEDomains((Ω,Γ),(Ω,Ω_out,Γ,Γg))
# _test = TProductFESpace(Ωbg,reffe,conformity=:H1;dirichlet_tags=["boundary"])
# _trial = ParamTrialFESpace(_test,g0μ)
# _feop = LinearParamFEOperator(b,_a,pspace,_trial,_test,_domains)

# _fesnaps,_festats = solution_snapshots(rbsolver,_feop)

# using Gridap.CellData
# dof_map_Ω = change_domain(feop.dof_maps.dof_map,Ω)
# _fesnaps_Ω = Snapshots(_fesnaps.data,dof_map_Ω,_fesnaps.realization)

# u1 = flatten_snapshots(fesnaps)[:,1]
# r1 = get_realization(fesnaps)[1]
# U1 = param_getindex(trial(r1),1)
# uh = FEFunction(U1,u1)
# writevtk(Ω,datadir("plts/sol"),cellfields=["uh"=>uh])

# _u1 = flatten_snapshots(_fesnaps)[:,1]
# _U1 = param_getindex(_trial(r1),1)
# _uh = FEFunction(_U1,_u1)
# writevtk(Ω,datadir("plts/sol_wrong_Ω"),cellfields=["uh"=>_uh])
# writevtk(Ωbg.trian,datadir("plts/sol_wrong_Ωbg"),cellfields=["uh"=>_uh])
