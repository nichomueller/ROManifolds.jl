using Gridap
using GridapEmbedded
using ROManifolds
using ROManifolds.DofMaps
using ROManifolds.TProduct
using DrWatson
using Test

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 10
partition = (n,n)

dp = pmax - pmin

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

model = CartesianDiscreteModel(pmin,pmax,partition)
cutgeo = cut(model,geo2)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ωbg = Triangulation(model)
Ωact = Triangulation(cutgeo,ACTIVE)
Ωactout = Triangulation(cutgeo,ACTIVE_OUT)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)

order = 2
degree = 2*order

dΩ = Measure(Ω,degree)
dΩout = Measure(Ωactout,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

ν(μ) = x->μ[3]
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

g(μ) = x->0
gμ(μ) = ParamFunction(g,μ)

const γd = 10.0
const hd = dp[1]/n

a(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(u,v,dΩ,dΓ,dΓn) =  a(u,v,dΩ,dΓ) - ( ∫(f⋅v)dΩ + ∫(h⋅v)dΓn + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ )

trian_a = (Ω,Γ)
trian_l = (Ω,Γ,Γn)
domains = FEDomains(trian_l,trian_a)

reffe = ReferenceFE(lagrangian,Float64,order)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)
Uagg = ParamTrialFESpace(Vagg,gμ)

feop = LinearParamOperator(l,a,pspace,Uagg,Vagg,domains)

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(μ,v) = ∫(∇(v)⋅∇(gμ(μ)))dΩout
Vout = FESpace(Ωactout,reffe,conformity=:H1)
Uout = ParamTrialFESpace(Vout,gμ)

V = FESpace(model,reffe,conformity=:H1)

agg_dof_to_bgdof = get_dof_to_bg_dof(V,Vagg)
act_out_dof_to_bgdof = get_dof_to_bg_dof(V,Vout)
agg_out_dof_to_bgdof = setdiff(act_out_dof_to_bgdof,agg_dof_to_bgdof)
agg_out_dof_to_act_out_dof = get_bg_dof_to_dof(V,Vout,agg_out_dof_to_bgdof)

ext = HarmonicExtension(A,b,agg_out_dof_to_bgdof,agg_out_dof_to_act_out_dof,aout,lout,μ)
solver = ExtensionSolver(LUSolver(),ext)
u = solve(solver,op.op)
