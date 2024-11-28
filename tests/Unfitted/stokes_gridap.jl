using Gridap
using GridapEmbedded
using Test

u(x) = VectorValue(x[1]*x[1], x[2])
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

eu = u - uh
ep = p - ph

l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))

eu_l2 = l2(eu)
eu_h1 = h1(eu)
ep_l2 = l2(ep)

# no aggregation, artificial viscosity instead

Ω_all = Triangulation(bgmodel)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
dΩ_out = Measure(Ω_out,degree)
V = FESpace(Ω_all,reffe_u,conformity=:H1)
Q = FESpace(Ω_all,reffe_p;conformity=:L2)
U = TrialFESpace(V)
P = TrialFESpace(Q)
Y = MultiFieldFESpace([V,Q];style=BlockMultiFieldStyle())
X = MultiFieldFESpace([U,P];style=BlockMultiFieldStyle())
a((u,p),(v,q)) =
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ +
  ∫( 1e-3*(∇(v)⊙∇(u) + q*p) ) * dΩ_out +
  ∫( (γ/h)*v⋅u - v⋅(n_Γd⋅∇(u)) - (n_Γd⋅∇(v))⋅u + (p*n_Γd)⋅v + (q*n_Γd)⋅u ) * dΓd
l((v,q)) =
  ∫( v⋅f - q*g ) * dΩ +
  ∫( (γ/h)*v⋅u - (n_Γd⋅∇(v))⋅u + (q*n_Γd)⋅u ) * dΓd +
  ∫( v⋅(n_Γn⋅∇(u)) - (n_Γn⋅v)*p ) * dΓn
op = AffineFEOperator(a,l,X,Y)
uh,ph = solve(op)

writevtk(Ω,"data/plts/sol",cellfields=["uh"=>uh,"ph"=>ph])

eu = u - uh
ep = p - ph

l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))

eu_l2 = l2(eu)
eu_h1 = h1(eu)
ep_l2 = l2(ep)

# parametric version, no tproduct

using ReducedOrderModels
using Gridap.MultiField

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)

test_u = FESpace(Ω,reffe_u,conformity=:H1)
trial_u = TrivialParamFESpace(test_u,5)

test_p = FESpace(Ω,reffe_p;conformity=:L2)
trial_p = TrialFESpace(test_p)

uμfun(μ,x) = VectorValue(μ[1]*x[1]*x[1],μ[2]*x[2])
uμfun(μ) = x -> uμfun(μ,x)
uμ(μ) = ParamFunction(uμfun,μ)
pμfun(μ,x) = μ[3]*(x[1] - x[2])
pμfun(μ) = x -> pμfun(μ,x)
pμ(μ) = ParamFunction(pμfun,μ)

fμfun(μ,x) = - Δ(uμfun(μ))(x) + ∇(pμfun(μ))(x)
fμfun(μ) = x -> fμfun(μ,x)
fμ(μ) = ParamFunction(fμfun,μ)
gμfun(μ,x) = (∇⋅uμfun(μ))(x)
gμfun(μ) = x -> gμfun(μ,x)
gμ(μ) = ParamFunction(gμfun,μ)

aμ(μ,(u,p),(v,q)) =
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ_in +
  ∫( 1e-3*(∇(v)⊙∇(u) + p*q) ) * dΩ_out +
  ∫( (γ/h)*v⋅u - v⋅(n_Γd⋅∇(u)) - (n_Γd⋅∇(v))⋅u + (p*n_Γd)⋅v + (q*n_Γd)⋅u ) * dΓd

lμ(μ,(u,p),(v,q)) =
  ∫( v⋅fμ(μ) - q*gμ(μ) ) * dΩ_in +
  ∫( (γ/h)*v⋅uμ(μ) - (n_Γd⋅∇(v))⋅uμ(μ) + (q*n_Γd)⋅uμ(μ) ) * dΓd +
  ∫( v⋅(n_Γn⋅∇(uμ(μ))) - (n_Γn⋅v)*pμ(μ) ) * dΓn -
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ_in

test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(lμ,aμ,pspace,trial,test)
fesolver = LinearFESolver(LUSolver())

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

tol = 1e-4
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=5,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=5,nparams_jac=5)

fesnaps,festats = solution_snapshots(rbsolver,feop)

μ = realization(feop;nparams=5)
aa((u,p),(v,q)) = aμ(μ,(u,p),(v,q))
ll((u,p),(v,q)) = lμ(μ,(u,p),(v,q))
du,dp = get_trial_fe_basis(test)
uh0,ph0 = zero(trial(μ))
v,q = get_fe_basis(test)
L = assemble_vector(ll((uh0,ph0),(v,q)),test)
A = assemble_matrix(aa,test,test)

# plotting

r = get_realization(fesnaps)
r1 = r[1]
U1 = param_getindex(trial_u(r1),1)
P1 = trial_p(r1)
uh1 = FEFunction(U1,fesnaps[1][:,1])
ph1 = FEFunction(P1,fesnaps[2][:,1])
writevtk(Ω_in,"data/plts/sol_u.vtu",cellfields=["uh"=>uh1])
writevtk(Ω_in,"data/plts/sol_p.vtu",cellfields=["ph"=>ph1])
