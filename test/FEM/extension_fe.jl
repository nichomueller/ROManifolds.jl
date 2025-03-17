using Gridap
using Gridap.Arrays
using Gridap.FESpaces
using GridapEmbedded
using ROManifolds
using ROManifolds.Extensions
using ROManifolds.DofMaps
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

order = 1
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωout,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

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

Vext = HarmonicExtensionFESpace(V,Vagg,Voutagg,aout,lout)

# get_cell_dof_ids(Vext,Ωact)
# get_cell_dof_ids(Vext,Ωactout)

# Vext[1] === get_internal_space(Vext)
# Vext[2] === get_external_space(Vext)
# nintext = num_free_dofs(get_internal_space(Vext)) + num_free_dofs(get_external_space(Vext))
# num_free_dofs(Vext) == nintext

# Extensions.get_extension(Vext)

# zero_dirichlet_values(Vext)
# get_dof_value_type(Vext)
# get_free_dof_ids(Vext)
# ConstraintStyle(Vext)
# get_fe_basis(Vext)

# get_cell_isconstrained(Vext,Ωact)
# get_cell_isconstrained(Vext,Ωactout)

# Extensions.to_multi_field(Vext)

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers

Uext = Vext # this needs fixing
bblocks = [LinearSystemBlock(),LinearSystemBlock()]
solver = BlockDiagonalSolver(bblocks,[LUSolver(),LUSolver()])

const γd = 10.0
const hd = dp[1]/n

a((ui,uo),(vi,vo)) = ∫(∇(vi)⋅∇(ui))dΩ + ∫( (γd/hd)*vi*ui  - vi*(n_Γ⋅∇(ui)) - (n_Γ⋅∇(vi))*ui )dΓ
l((vi,vo)) = ∫( (γd/hd)*vi*g - (n_Γ⋅∇(vi))*g )dΓ

op = AffineFEOperator(a,l,Uext,Vext)
A = get_matrix(op)
b = get_vector(op)

uhin,uhex = solve(op)

writevtk(Ωbg,"sol_inout",cellfields=["uhin"=>uhin,"uhex"=>uhex])

# _Vaggext = ExternalFESpace(V,Vagg,Voutact,aggregatesout)

# Aout = assemble_matrix(aout,Voutagg,Voutagg)
# _Aout = assemble_matrix(aout,_Vaggext,_Vaggext)
# Aout ≈ _Aout

ν(μ) = x->μ[3]
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

h(μ) = x->1
hμ(μ) = ParamFunction(h,μ)

g(μ) = x->μ[3]*x[1]-x[2]
gμ(μ) = ParamFunction(g,μ)

a(μ,(u,uo),(v,vo),dΩ,dΓ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(μ,(v,vo),dΩ,dΓ,dΓn) = ∫(fμ(μ)⋅v)dΩ + ∫(hμ(μ)⋅v)dΓn + ∫( (γd/hd)*v*gμ(μ) - (n_Γ⋅∇(v))*gμ(μ) )dΓ
res(μ,u,v,dΩ,dΓ,dΓn) =  a(μ,u,v,dΩ,dΓ) - l(μ,v,dΩ,dΓ,dΓn)

trian_a = (Ω,Γ)
trian_res = (Ω,Γ,Γn)
domains = FEDomains(trian_res,trian_a)

aout(μ,u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(μ,v) = ∫(∇(v)⋅∇(gμ(μ)))dΩout

μ = realization(pspace;nparams=50)
ext = HarmonicExtension(Voutagg,aout,lout,μ)

Vext = Extensions.SingleFieldExtensionFESpace(ext,V,Vagg,Voutagg)
Uext = ParamTrialFESpace(Vext,gμ)

feop = LinearParamOperator(res,a,pspace,Uext,Vext,domains)
