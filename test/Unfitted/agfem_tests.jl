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
n = 4
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

h(μ) = x->1
hμ(μ) = ParamFunction(h,μ)

g(μ) = x->μ[3]*x[1]-x[2]
gμ(μ) = ParamFunction(g,μ)

const γd = 10.0
const hd = dp[1]/n

a(μ,u,v,dΩ,dΓ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(μ,v,dΩ,dΓ,dΓn) = ∫(fμ(μ)⋅v)dΩ + ∫(hμ(μ)⋅v)dΓn + ∫( (γd/hd)*v*gμ(μ) - (n_Γ⋅∇(v))*gμ(μ) )dΓ
res(μ,u,v,dΩ,dΓ,dΓn) =  a(μ,u,v,dΩ,dΓ) - l(μ,v,dΩ,dΓ,dΓn)

trian_a = (Ω,Γ)
trian_res = (Ω,Γ,Γn)
domains = FEDomains(trian_res,trian_a)

reffe = ReferenceFE(lagrangian,Float64,order)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)
Uagg = ParamTrialFESpace(Vagg,gμ)

feop = LinearParamOperator(res,a,pspace,Uagg,Vagg,domains)

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(μ,v) = ∫(∇(v)⋅∇(gμ(μ)))dΩout
Vout = FESpace(Ωactout,reffe,conformity=:H1)
Uout = ParamTrialFESpace(Vout,gμ)

V = FESpace(model,reffe,conformity=:H1)

agg_dof_to_bgdof = get_dof_to_bg_dof(V,Vagg)
act_out_dof_to_bgdof = get_dof_to_bg_dof(V,Vout)
agg_out_dof_to_bgdof = setdiff(act_out_dof_to_bgdof,agg_dof_to_bgdof)
agg_out_dof_to_act_out_dof = get_bg_dof_to_dof(V,Vout,agg_out_dof_to_bgdof)

using ROManifolds.ParamSteady
using ROManifolds.Utils

μ = realization(feop;nparams=2)
ext = HarmonicExtension(V,Vagg,Uout,Vout,aout,lout,μ)
solver = ExtensionSolver(LUSolver(),ext)
nlop = parameterize(set_domains(feop),μ)
u = solve(solver,nlop)

act_odofs_ids = DofMaps.get_cell_odof_ids(Vout)
odofs_ids = DofMaps.get_cell_odof_ids(V)
dof_to_odof = DofMaps.reorder_dofs(V,odofs_ids)

# agg_dof_to_bg_odof = DofMaps.get_odof_to_bg_odof(V,Vagg)
# act_out_dof_to_bg_odof = get_dof_to_bg_dof(V,Vout)
# agg_out_dof_to_bg_odof = setdiff(act_out_dof_to_bgdof,agg_dof_to_bgdof)
# agg_out_dof_to_act_out_dof = get_bg_dof_to_dof(V,Vout,agg_out_dof_to_bgdof)


# dof tests

agg_dof_to_bgdof = get_dof_to_bg_dof(V,Vagg)
bgdof_to_agg_dof = get_bg_dof_to_dof(V,Vagg)

agg_dof_to_bgdof = get_dof_to_bg_dof(V,Vagg)
agg_dof_to_bg_odof = DofMaps.reorder_dof_map(dof_to_odof,agg_dof_to_bgdof)
act_out_dof_to_bgdof = get_dof_to_bg_dof(V,Vout)
act_out_dof_to_bg_odof = DofMaps.reorder_dof_map(dof_to_odof,act_out_dof_to_bgdof)
agg_out_dof_to_bgdof = setdiff(act_out_dof_to_bgdof,agg_dof_to_bgdof)
agg_out_dof_to_bg_odof = setdiff(act_out_dof_to_bg_odof,agg_dof_to_bg_odof)

bg_dof_to_act_out_dof = get_bg_dof_to_dof(V,Vout)

@assert intersect(act_out_dof_to_bgdof,agg_out_dof_to_bgdof) == agg_out_dof_to_bgdof
@assert intersect(act_out_dof_to_bg_odof,agg_out_dof_to_bg_odof) == agg_out_dof_to_bg_odof

agg_out_dof_to_act_out_odof = zeros(Int32,length(agg_out_dof_to_bg_odof))
for (iagg,bg_odof) in enumerate(agg_out_dof_to_bg_odof)
  act_odof = findfirst(act_out_dof_to_bg_odof.==bg_odof)
  agg_out_dof_to_act_out_odof[iagg] = act_odof
end

_μ = [1,2,3]
_lout(v) = ∫(∇(v)⋅∇(g(_μ)))dΩout
cell_odofs_ids = DofMaps.get_cell_odof_ids(Vout)
oVout = CartesianFESpace(Vout,cell_odofs_ids,[1,],[1,])
A = assemble_matrix(aout,oVout,oVout)
b = assemble_vector(_lout,oVout)

ext = HarmonicExtension(A,b,agg_out_dof_to_bg_odof,agg_out_dof_to_act_out_odof)
solver = ExtensionSolver(LUSolver(),ext)

_a(u,v) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
_l(v) =  ∫(f(_μ)⋅v)dΩ + ∫(h(_μ)⋅v)dΓn + ∫( (γd/hd)*v*g(_μ) - (n_Γ⋅∇(v))*g(_μ) )dΓ

op = AffineFEOperator(_a,_l,Vagg,oVout)
u = solve(solver,op.op)
uh = FEFunction(V,u)
