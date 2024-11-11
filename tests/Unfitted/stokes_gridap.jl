using Gridap
using GridapEmbedded
using Gridap.MultiField

const L = 1
const R  = 0.1
const n = 30

const domain = (0,L,0,L)
const partition = (n,n)

p1 = Point(0.3,0.5)
p2 = Point(0.7,0.5)
geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = !union(geo1,geo2)

bgmodel = CartesianDiscreteModel(domain,partition)
cutgeo = cut(bgmodel,geo3)

Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γd = EmbeddedBoundary(cutgeo)
Γn = BoundaryTriangulation(bgmodel;tags="boundary")

Ω_act = Triangulation(cutgeo,ACTIVE)
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

nΓ_inlet = get_normal_vector(Γ_inlet)
nΓ_outlet = get_normal_vector(Γ_outlet)
nΓ_in = get_normal_vector(Γ_in)

order = 2
degree = 2*order

dΩ_in = Measure(Ω_in,degree)

dΓ_in = Measure(Γ_in,degree)
dΓ_inlet = Measure(Γ_inlet,degree)

g_in(x) = VectorValue(-x[2]*(1.0-x[2]),0.0)
g_0(x) = VectorValue(0.0,0.0)

γd = 10    # Nitsche coefficient (Dirichlet)
γn = 10    # Nitsche coefficient (Neumann)
h = L/n     # Mesh size

nitsche_jac(u,v,dΓ,nΓ) = ∫( (γd/h)*v⋅u  - v⋅(nΓ⋅∇(u)) - (nΓ⋅∇(v))⋅u )dΓ
nitsche_res(v,dΓ,nΓ) = ∫( g_in ⋅ ((γd/h)*v  - (nΓ⋅∇(v))) )dΓ

a11(u,v) = ∫( ∇(v)⊙∇(u) )dΩ_in + nitsche_jac(u,v,dΓ_in,nΓ_in) + nitsche_jac(u,v,dΓ_inlet,nΓ_inlet)
jac((u,p),(v,q)) = a11(u,v) - ∫(p*(∇⋅(v)))dΩ_in + ∫(q*(∇⋅(u)))dΩ_in
res((v,q)) = ∫( g_in ⋅ ((γd/h)*v  - (nΓ_inlet⋅∇(v))) )dΓ_inlet

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = AgFEMSpace(TestFESpace(Ω_act,reffe_u,conformity=:H1,dirichlet_tags="wall"),aggregates)
trial_u = TrialFESpace(test_u,g_0)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = AgFEMSpace(TestFESpace(Ω_act,reffe_p;conformity=:C0),aggregates)
trial_p = TrialFESpace(test_p)
test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = AffineFEOperator(jac,res,trial,test)

fesolver = LinearFESolver(LUSolver())

uh = solve(fesolver,feop)


############## ok version

using Gridap
using GridapEmbedded
using Test

u(x) = VectorValue(x[1]*x[1], x[2])
u0(x) = VectorValue(0.0, 0.0)
p(x) = x[1] - x[2]

f(x) = - Δ(u)(x) + ∇(p)(x)
g(x) = (∇⋅u)(x)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 20
partition = (n,n)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
dp = pmax - pmin
const h = dp[1]/n

cutgeo = cut(bgmodel,geo2)
Ω_act = Triangulation(cutgeo,ACTIVE)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ω = Triangulation(cutgeo,PHYSICAL)
Γd = EmbeddedBoundary(cutgeo)
Γn = BoundaryTriangulation(Ω_act,tags="boundary")
order = 2
degree = 2*order
dΩ = Measure(Ω,degree)
dΓd = Measure(Γd,degree)
dΓn = Measure(Γn,degree)

n_Γd = get_normal_vector(Γd)
n_Γn = get_normal_vector(Γn)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
Vstd = FESpace(Ω_act,reffe_u,conformity=:H1)

reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)
Qstd = FESpace(Ω_act,reffe_p;conformity=:L2)

V = AgFEMSpace(Vstd,aggregates)
Q = AgFEMSpace(Qstd,aggregates)

U = TrialFESpace(V)
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V,Q])
X = MultiFieldFESpace([U,P])

const γ = order*(order+1)

a((u,p),(v,q)) =
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ +
  ∫( (γ/h)*v⋅u - v⋅(n_Γd⋅∇(u)) - (n_Γd⋅∇(v))⋅u + (p*n_Γd)⋅v + (q*n_Γd)⋅u ) * dΓd

l((v,q)) =
  ∫( v⋅f - q*g ) * dΩ +
  ∫( (γ/h)*v⋅u - (n_Γd⋅∇(v))⋅u + (q*n_Γd)⋅u ) * dΓd +
  ∫( v⋅(n_Γn⋅∇(u)) - (n_Γn⋅v)*p ) * dΓn

op = AffineFEOperator(a,l,X,Y)
uh,ph = solve(op)
writevtk(Ω,"trian_O",cellfields=["uh"=>uh,"ph"=>ph])

# try tweaking the boundary conditions

u(x) = VectorValue(-x[2]*(1-x[2]), 0.0)
u0(x) = VectorValue(0.0, 0.0)

labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"wall",collect(1:6))
add_tag_from_tags!(labels,"inlet",[7])
add_tag_from_tags!(labels,"outlet",[8])

Γ = EmbeddedBoundary(cutgeo)
Γd = BoundaryTriangulation(Ω_act,tags="inlet")
Γn = BoundaryTriangulation(Ω_act,tags="outlet")

dΓ = Measure(Γ,degree)
dΓd = Measure(Γd,degree)
dΓn = Measure(Γn,degree)
n_Γ = get_normal_vector(Γ)
n_Γd = get_normal_vector(Γd)
n_Γn = get_normal_vector(Γn)

Vstd = FESpace(Ω_act,reffe_u,conformity=:H1;dirichlet_tags="wall")
V = AgFEMSpace(Vstd,aggregates)

U = TrialFESpace(V)
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V,Q])
X = MultiFieldFESpace([U,P])

a((u,p),(v,q)) =
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ +
  ∫( (γ/h)*v⋅u - v⋅(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))⋅u + (p*n_Γ)⋅v + (q*n_Γ)⋅u ) * dΓ +
  ∫( (γ/h)*v⋅u - v⋅(n_Γd⋅∇(u)) - (n_Γd⋅∇(v))⋅u + (p*n_Γd)⋅v + (q*n_Γd)⋅u ) * dΓd

l((v,q)) =
  ∫( v⋅f - q*g ) * dΩ +
  ∫( (γ/h)*v⋅u0 - (n_Γ⋅∇(v))⋅u0 + (q*n_Γ)⋅u0 ) * dΓ +
  ∫( (γ/h)*v⋅u - (n_Γd⋅∇(v))⋅u + (q*n_Γd)⋅u ) * dΓd +
  ∫( v⋅(n_Γn⋅∇(u)) - (n_Γn⋅v)*p ) * dΓn

op = AffineFEOperator(a,l,X,Y)
uh,ph = solve(op)
writevtk(Ω,"trian_O",cellfields=["uh"=>uh,"ph"=>ph])
