using Gridap
using Gridap.FESpaces
using GridapEmbedded
using ROManifolds
using ROManifolds.DofMaps
using ROManifolds.TProduct
using ROManifolds.Utils
using ROManifolds.ParamAlgebra
using ROManifolds.ParamSteady
using ROManifolds.RBSteady
using SparseArrays
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

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωout,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

reffe = ReferenceFE(lagrangian,Float64,1)

Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)

cutgeoout = cut(model,!geo2)
aggregatesout = aggregate(strategy,cutgeoout)
Voutact = FESpace(Ωactout,reffe,conformity=:H1)
Voutagg = AgFEMSpace(Voutact,aggregatesout)

g(x) = x[2]-x[1]

Uagg = TrialFESpace(Vagg,g)

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(v) = ∫(∇(v)⋅∇(g))dΩout

V = FESpace(model,reffe,conformity=:H1)

extfe = HarmonicExtensionFESpace(V,Vagg,Voutagg,aout,lout)

ν(μ) = x->μ[3]
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

h(μ) = x->1
hμ(μ) = ParamFunction(h,μ)

g(μ) = x->μ[3]*x[1]-x[2]
gμ(μ) = ParamFunction(g,μ)

gg(x) = x[1]

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
Uagg = TrialFESpace(Vagg,gg)

feop = LinearParamOperator(res,a,pspace,Uagg,Vagg,domains)

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(v) = ∫(∇(v)⋅∇(gg))dΩout
Vout = FESpace(Ωactout,reffe,conformity=:H1)
Uout = TrialFESpace(Vout,gg)

V = FESpace(model,reffe,conformity=:H1)
U = TrialFESpace(V,gg)

ext = HarmonicExtension(V,Vagg,Uout,Vout,aout,lout)

laplacian = assemble_matrix(aout,Uout,Vout)
residual = assemble_vector(lout,Vout)

laplacianin = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩ,Vact,Vact)
laplacianout = assemble_matrix((u,v)->∫(∇(v)⋅∇(u))dΩout,Vout,Vout)

uh = interpolate_everywhere(x->x[2]-x[1],Vout)
writevtk(Ωactout,"sol_out",cellfields=["uh"=>uh])
writevtk(Ωbg,"sol",cellfields=["uh"=>uh])

cutgeoout = cut(model,!geo2)
aggregatesout = aggregate(strategy,cutgeoout)
Voutagg = AgFEMSpace(Vout,aggregatesout)

ain(u,v) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
lin(v) = ∫(v)dΩ
aout(u,v) = ∫(∇(v)⋅∇(u))dΩout + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
lout(v) = ∫(v)dΩout

opin = AffineFEOperator(ain,lin,Vagg,Vagg)
opout = AffineFEOperator(aout,lout,Voutagg,Voutagg)

uin = solve(opin)
uout = solve(opout)

writevtk(Ωbg,"sols",cellfields=["uin"=>uin,"uout"=>uout])
writevtk(Ω,"sol_in",cellfields=["uin"=>uin])
writevtk(Ωout,"sol_out",cellfields=["uout"=>uout])

# using Gridap.ReferenceFEs
# using Gridap.Geometry
# xin = unique(get_node_coordinates(Ω))
# xout = unique(get_node_coordinates(Ωout))

# intersect(xin,xout)

num_free_dofs(V)
num_free_dofs(Voutagg)
num_free_dofs(Vagg)

get_node_coordinates(get_triangulation(V))

#
R = 0.5*(1+sqrt(2)+1e-6)
geo = disk(R,x0=Point(0,0))
pmin = Point(-2.5,-2.5)
pmax = Point(2.5,2.5)
partition = (5,5)
model = CartesianDiscreteModel(pmin,pmax,partition)
cutgeo = cut(model,!geo)
Ωbg = Triangulation(model)
Ωact = Triangulation(cutgeo,ACTIVE)
Ωactout = Triangulation(cutgeo,ACTIVE_OUT)
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)
reffe = ReferenceFE(lagrangian,Float64,2)
V = FESpace(Ωbg,reffe,conformity=:H1)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)
cutgeoout = cut(model,geo)
aggregatesout = aggregate(strategy,cutgeoout)
Vout = FESpace(Ωactout,reffe,conformity=:H1)
Voutagg = AgFEMSpace(Vout,aggregatesout)
num_free_dofs(V)
num_free_dofs(Vagg)
num_free_dofs(Voutagg)
