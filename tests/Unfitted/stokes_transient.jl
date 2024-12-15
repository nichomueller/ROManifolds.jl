using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ReducedOrderModels

θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 60*dt

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

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

bgmodel = TProductModel(domain,partition)
cutgeo = cut(bgmodel,geo3)

labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"wall",collect(1:7))
add_tag_from_tags!(labels,"Γn",[8])

Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γn = BoundaryTriangulation(bgmodel;tags="Γn")
Γd = EmbeddedBoundary(cutgeo)
n_Γn = get_normal_vector(Γn)
n_Γd = get_normal_vector(Γd)

order = 2
degree = 2*order

dΩ = Measure(Ω,degree)
dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)
dΓn = Measure(Γn,degree)
dΓd = Measure(Γd,degree)

u(x,μ,t) = VectorValue(x[1]*x[1],x[2])*abs(μ[1]/μ[2]*sin(μ[3]*t))
u(μ,t) = x->u(x,μ,t)
uμt(μ,t) = TransientParamFunction(u,μ,t)

p(x,μ,t) = abs((x[1]-x[2])*(1-cos(μ[3]*t)))
p(μ,t) = x->p(x,μ,t)
pμt(μ,t) = TransientParamFunction(p,μ,t)

f(x,μ,t) = - Δ(u(μ,t))(x) + ∇(p(μ,t))(x)
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = (∇⋅u(μ,t))(x)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

g_0(x,μ,t) = VectorValue(0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

const ν0 = 1e-3
const γ = order*(order+1)
const h = L/n

a(μ,t,(u,p),(v,q),dΩ_in,dΩ_out,dΓd) =
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p )dΩ_in +
  ∫( ν0*(∇(v)⊙∇(u) - q*p) )dΩ_out +
  ∫( (γ/h)*v⋅u - v⋅(n_Γd⋅∇(u)) - (n_Γd⋅∇(v))⋅u + (p*n_Γd)⋅v + (q*n_Γd)⋅u )dΓd

res(μ,t,(u,p),(v,q),dΩ_in,dΓd,dΓn) =
  ∫( v⋅fμt(μ,t) - q*gμt(μ,t) )dΩ_in +
  ∫( (γ/h)*v⋅uμt(μ,t) - (n_Γd⋅∇(v))⋅uμt(μ,t) + (q*n_Γd)⋅uμt(μ,t) )dΓd +
  ∫( v⋅(n_Γn⋅∇(uμt(μ,t))) - (n_Γn⋅v)*pμt(μ,t) )dΓn -
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p )dΩ_in

m(μ,t,(uₜ,pₜ),(v,q),dΩ_in) = ∫(v⋅uₜ)dΩ_in

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

trian_res = (Ω_in,Γd,Γn)
trian_a = (Ω_in,Ω_out,Γd)
trian_m = (Ω_in,)
domains = FEDomains(trian_res,(trian_a,trian_m))

coupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags="wall")
trial_u = TransientTrialParamFESpace(test_u,uμt)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:C0)
trial_p = TrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((a,m),res,ptspace,trial,test,domains)

fesolver = ThetaMethod(LUSolver(),dt,θ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

tol = fill(1e-4,5)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=20)

fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ;r)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10,random=true)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon,xh0μ)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

r = μon
S′ = flatten_snapshots(fesnaps)
S1 = S′[1][:,:,1]
r1 = r[1,:]
U1 = trial_u(r1)
plt_dir = datadir("plts")
create_dir(plt_dir)
for i in 1:length(r1)
  Ui = param_getindex(U1,i)
  uhi = FEFunction(Ui,S1[:,i])
  writevtk(Ω_in,joinpath(plt_dir,"u_$i.vtu"),cellfields=["uh"=>uhi])
end
S2 = S′[2][:,:,1]
for i in 1:length(r1)
  Pi = trial_p
  phi = FEFunction(Pi,S2[:,i])
  writevtk(Ω_in,joinpath(plt_dir,"p_$i.vtu"),cellfields=["ph"=>phi])
end

# GRIDAP

using Gridap
using Gridap.ODEs
using GridapEmbedded

θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 20*dt

ut(t) = x -> VectorValue(x[1]*x[1],x[2])*t
u = TimeSpaceFunction(ut)
pt(t) = x -> (x[1] - x[2])*t
p = TimeSpaceFunction(pt)

ft(t) = x -> ∂t(u)(t,x) - Δ(u)(t,x) + ∇(p)(t,x)
f = TimeSpaceFunction(ft)
g = ∇⋅u

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

Ω_bg = Triangulation(bgmodel)
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

model = get_active_model(Ω_act)
V_cell_fe_std = FiniteElements(PhysicalDomain(),
                               model,
                               lagrangian,
                               VectorValue{2,Float64},
                               order)
Vstd = FESpace(Ω_act,V_cell_fe_std)

V_cell_fe_ser = FiniteElements(PhysicalDomain(),
                               model,
                               lagrangian,
                               VectorValue{2,Float64},
                               order,
                               space=:S,
                               conformity=:L2)
# RMK: we don't neet to impose continuity since
# we only use the cell dof basis / shapefuns
Vser = FESpace(Ω_act,V_cell_fe_ser)

Q_cell_fe_std = FiniteElements(PhysicalDomain(),
                               model,
                               lagrangian,
                               Float64,
                               order-1,
                               space=:P,
                               conformity=:L2)
Qstd = FESpace(Ω_act,Q_cell_fe_std)

V = AgFEMSpace(Vstd,aggregates,Vser)
Q = AgFEMSpace(Qstd,aggregates)

U = TransientTrialFESpace(V,u)
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V,Q])
X = MultiFieldFESpace([U,P])

const γ = order*(order+1)

a(t,(u,p),(v,q)) =
  ∫( ∇(v)⊙∇(u) - (∇⋅v)*p - q*(∇⋅u) )*dΩ +
  ∫( (γ/h)*v⋅u - v⋅(n_Γd⋅∇(u)) - (n_Γd⋅∇(v))⋅u + (p*n_Γd)⋅v + (q*n_Γd)⋅u )*dΓd

l(t,(v,q)) =
  ∫( v⋅f(t) )*dΩ - ∫( q*g(t) )*dΩ +
  ∫( (γ/h)*v⋅u(t) )*dΓd - ∫( (n_Γd⋅∇(v))⋅u(t) )*dΓd + ∫( (q*n_Γd)⋅u(t) )*dΓd +
  ∫( v⋅(n_Γn⋅∇(u(t))) )*dΓn - ∫( (n_Γn⋅v)*p(t) )*dΓn

m(t,(uₜ,pₜ),(v,q)) = ∫(v⋅uₜ)dΩ

l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))

op = TransientLinearFEOperator((a,m),l,X,Y)

odeslvr = ThetaMethod(LUSolver(),dt,θ)

X0 = X(t0)
_xh0 = interpolate_everywhere([u(t0),p(t0)],X0)
(_,xh0),= solve(odeslvr,op,t0,dt,_xh0)

xht = solve(odeslvr,op,t0,tf,xh0)

plt_dir = datadir("plts")
create_dir(plt_dir)

UU,PP = Vector{Float64}[],Vector{Float64}[]
for (t,xh) in xht
  uh,ph = xh
  push!(UU,copy(uh.free_values))
  push!(PP,copy(ph.free_values))
  writevtk(Ω_bg,joinpath(plt_dir,"sol_Ω_$t.vtu"),cellfields=["uh"=>uh,"ph"=>ph])
  writevtk(Ω,joinpath(plt_dir,"sol_Ωin_$t.vtu"),cellfields=["uh"=>uh,"ph"=>ph])

  println("err H1(u): $(h1(u(t) - uh))")
  println("err L2(p): $(l2(p(t) - ph))")
end

# with different fe spaces

Ω_hole = Triangulation(cutgeo,PHYSICAL_OUT)
dΩ_hole = Measure(Ω_hole,degree)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)

V′ = TestFESpace(Ω_bg,reffe_u;conformity=:H1,dirichlet_tags="boundary")
Q′ = TestFESpace(Ω_bg,reffe_p;conformity=:L2)

U′ = TransientTrialFESpace(V′,u)
P′ = TrialFESpace(Q′)

Y′ = MultiFieldFESpace([V′,Q′])
X′ = MultiFieldFESpace([U′,P′])

a′(t,(u,p),(v,q)) =
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p )*dΩ +
  ∫( 1e-3*∇(v)⊙∇(u) - 1e-3*q*p )*dΩ_hole +
  ∫( (γ/h)*v⋅u - v⋅(n_Γd⋅∇(u)) - (n_Γd⋅∇(v))⋅u + (p*n_Γd)⋅v + (q*n_Γd)⋅u )*dΓd

op′ = TransientLinearFEOperator((a′,m),l,X′,Y′)

X0′ = X′(t0)
xh0′ = interpolate_everywhere([u(t0),p(t0)],X0′)

xht′ = solve(odeslvr,op′,t0,tf,xh0′)

plt_dir = datadir("plts")
create_dir(plt_dir)

UU′,PP′ = Vector{Float64}[],Vector{Float64}[]
for (t,xh) in xht′
  uh,ph = xh
  push!(UU′,copy(uh.free_values))
  push!(PP′,copy(ph.free_values))
  writevtk(Ω_bg,joinpath(plt_dir,"_sol_Ω_$t.vtu"),cellfields=["uh"=>uh,"ph"=>ph])
  writevtk(Ω,joinpath(plt_dir,"_sol_Ωin_$t.vtu"),cellfields=["uh"=>uh,"ph"=>ph])

  println("err H1(u): $(h1(u(t) - uh))")
  println("err L2(p): $(l2(p(t) - ph))")
end

# what happens if I set 0 boundary condition with Nitsche? --> very bad solution

l′(t,(v,q)) =
  ∫( v⋅f(t) - q*g(t) )*dΩ +
  ∫( v⋅(n_Γn⋅∇(u(t))) - (n_Γn⋅v)*p(t) )*dΓn

op′ = TransientLinearFEOperator((a′,m),l′,X′,Y′)

X0′ = X′(t0)
xh0′ = interpolate_everywhere([u(t0),p(t0)],X0′)

xht′ = solve(odeslvr,op′,t0,tf,xh0′)

UU′,PP′ = Vector{Float64}[],Vector{Float64}[]
for (t,xh) in xht′
  uh,ph = xh
  push!(UU′,copy(uh.free_values))
  push!(PP′,copy(ph.free_values))
  writevtk(Ω_bg,joinpath(plt_dir,"_sol_Ω_$t.vtu"),cellfields=["uh"=>uh,"ph"=>ph])
  writevtk(Ω,joinpath(plt_dir,"_sol_Ωin_$t.vtu"),cellfields=["uh"=>uh,"ph"=>ph])
end

# ##############################THIS WORKS#######################################
# using Gridap.ODEs

# # Analytical functions
# ut(t) = x -> VectorValue(x[1],x[2])*t
# u = TimeSpaceFunction(ut)

# pt(t) = x -> (x[1] - x[2])*t
# p = TimeSpaceFunction(pt)

# # Geometry
# domain = (0,1,0,1)
# partition = (5,5)
# model = CartesianDiscreteModel(domain,partition)

# # FE spaces
# order = 2
# reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
# # V = FESpace(model,reffe_u,conformity=:H1,dirichlet_tags="boundary")
# # U = TransientTrialFESpace(V,u)
# V = FESpace(model,reffe_u,conformity=:H1)
# U = TransientTrialFESpace(V)

# reffe_p = ReferenceFE(lagrangian,Float64,order - 1)
# Q = FESpace(model,reffe_p,conformity=:H1,constraint=:zeromean)
# P = TrialFESpace(Q)

# X = TransientMultiFieldFESpace([U,P])
# Y = MultiFieldFESpace([V,Q])

# # Integration
# Ω = Triangulation(model)
# Γ = BoundaryTriangulation(Ω,tags="boundary")
# degree = 2*order
# dΩ = Measure(Ω,degree)
# dΓ = Measure(Γ,degree)
# n_Γ = get_normal_vector(Γ)

# γγ = order*(order+1)
# hh = 1/5

# # FE operator
# ft(t) = x -> ∂t(u)(t,x) - Δ(u)(t,x) + ∇(p)(t,x)
# f = TimeSpaceFunction(ft)
# g = ∇⋅u
# mass(t,∂ₜu,v) = ∫(∂ₜu⋅v)*dΩ
# stiffness(t,u,v) = ∫(∇(u) ⊙ ∇(v))*dΩ
# nitsche(t,(du,dp),(v,q)) = ∫( (γγ/hh)*v⋅du - v⋅(n_Γ⋅∇(du)) - (n_Γ⋅∇(v))⋅du + (dp*n_Γ)⋅v - (q*n_Γ)⋅du )*dΓ
# forcing(t,(v,q)) = ∫(f(t)⋅v)*dΩ + ∫(g(t)*q)*dΩ

# nres(t,(v,q)) = ∫( (γγ/hh)*v⋅u(t) )*dΓ - ∫( (n_Γ⋅∇(v))⋅u(t) )*dΓ - ∫( (q*n_Γ)⋅u(t) )*dΓ
# res(t,(v,q)) = forcing(t,(v,q)) + nres(t,(v,q))
# # res(t,(v,q)) = forcing(t,(v,q))
# # jac(t,(du,dp),(v,q)) = stiffness(t,du,v) - ∫(dp*(∇⋅v))*dΩ + ∫((∇⋅du)*q)*dΩ
# jac(t,(du,dp),(v,q)) = stiffness(t,du,v) + nitsche(t,(du,dp),(v,q)) - ∫(dp*(∇⋅v))*dΩ + ∫((∇⋅du)*q)*dΩ
# jac_t(t,(dut,dpt),(v,q)) = mass(t,dut,v)

# # Initial conditions
# t0 = 0.0
# tF = 1.0
# dt = 0.1

# U0 = U(t0)
# uh0 = interpolate_everywhere(u(t0),U0)
# P0 = P(t0)
# ph0 = interpolate_everywhere(p(t0),P0)
# X0 = X(t0)
# xh0 = interpolate_everywhere([uh0,ph0],X0)
# xhs0 = (xh0,)

# tfeop = TransientLinearFEOperator((jac,jac_t),res,X,Y)

# # ODE Solver
# using Test

# tol = 1.0e-6
# sysslvr_l = LUSolver()
# odeslvr = ThetaMethod(sysslvr_l,dt,0.5)
# fesltn = solve(odeslvr,tfeop,t0,tF*10,xhs0)
# for (t_n,xhs_n) in fesltn
#   println("err H1(u): $(h1(u(t_n) - xhs_n[1]))")
#   println("err L2(p): $(l2(p(t_n) - xhs_n[2]))")
# end

# ##############################################################################

using Gridap.ODEs

# Analytical functions
ut(t) = x -> VectorValue(x[1],x[2])*abs(sin(2pi*t/tF))
u = TimeSpaceFunction(ut)

pt(t) = x -> (x[1] - x[2])*abs(1-cos(2pi*t/tF))
p = TimeSpaceFunction(pt)

# Geometry
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

Ω_bg = Triangulation(bgmodel)
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

# FE spaces
order = 2
degree = 2*order
# reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
# V = FESpace(model,reffe_u,conformity=:H1)
# U = TransientTrialFESpace(V)
# reffe_p = ReferenceFE(lagrangian,Float64,order-1)
# Q = FESpace(model,reffe_p,conformity=:H1,constraint=:zeromean)
# P = TrialFESpace(Q)

model = get_active_model(Ω_act)
V_cell_fe_std = FiniteElements(PhysicalDomain(),
                               model,
                               lagrangian,
                               VectorValue{2,Float64},
                               order)
Vstd = FESpace(Ω_act,V_cell_fe_std)
V_cell_fe_ser = FiniteElements(PhysicalDomain(),
                               model,
                               lagrangian,
                               VectorValue{2,Float64},
                               order,
                               space=:S,
                               conformity=:L2)
Vser = FESpace(Ω_act,V_cell_fe_ser)
Q_cell_fe_std = FiniteElements(PhysicalDomain(),
                               model,
                               lagrangian,
                               Float64,
                               order-1,
                               space=:P,
                               conformity=:L2)
Qstd = FESpace(Ω_act,Q_cell_fe_std)
V = AgFEMSpace(Vstd,aggregates,Vser)
Q = AgFEMSpace(Qstd,aggregates)
U = TransientTrialFESpace(V,u)
P = TrialFESpace(Q)
Y = MultiFieldFESpace([V,Q])
X = MultiFieldFESpace([U,P])

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

γγ = order*(order+1)

# FE operator
ft(t) = x -> ∂t(u)(t,x) - Δ(u)(t,x) + ∇(p)(t,x)
f = TimeSpaceFunction(ft)
g = ∇⋅u
mass(t,∂ₜu,v) = ∫(∂ₜu⋅v)*dΩ
stiffness(t,u,v) = ∫(∇(u) ⊙ ∇(v))*dΩ
extension(t,(du,dp),(v,q)) = (∫(1e-3*∇(du) ⊙ ∇(v))*dΩ_hole + ∫(1e-3*dp*q)*dΩ_hole)
nitsche(t,(du,dp),(v,q)) = ∫( (γγ/h)*v⋅du - v⋅(n_Γd⋅∇(du)) - (n_Γd⋅∇(v))⋅du + (dp*n_Γd)⋅v - (q*n_Γd)⋅du )*dΓd
forcing(t,(v,q)) = ∫(f(t)⋅v)*dΩ + ∫(g(t)*q)*dΩ

nres(t,(v,q)) = ∫( (γγ/h)*v⋅u(t) )*dΓd - ∫( (n_Γd⋅∇(v))⋅u(t) )*dΓd - ∫( (q*n_Γd)⋅u(t) )*dΓd
res(t,(v,q)) = forcing(t,(v,q)) + nres(t,(v,q)) + ∫( v⋅(n_Γn⋅∇(u)(t)) - (n_Γn⋅v)*p(t) ) * dΓn
# jac(t,(du,dp),(v,q)) = stiffness(t,du,v) + nitsche(t,(du,dp),(v,q)) + extension(t,(du,dp),(v,q)) - ∫(dp*(∇⋅v))*dΩ + ∫((∇⋅du)*q)*dΩ
jac(t,(du,dp),(v,q)) = stiffness(t,du,v) + nitsche(t,(du,dp),(v,q)) - ∫(dp*(∇⋅v))*dΩ + ∫((∇⋅du)*q)*dΩ
jac_t(t,(dut,dpt),(v,q)) = mass(t,dut,v)

# Initial conditions
t0 = 0.0
tF = 1.0
dt = 0.1

U0 = U(t0)
uh0 = interpolate_everywhere(u(t0),U0)
P0 = P(t0)
ph0 = interpolate_everywhere(p(t0),P0)
X0 = X(t0)
xh0 = interpolate_everywhere([uh0,ph0],X0)
xhs0 = (xh0,)

tfeop = TransientLinearFEOperator((jac,jac_t),res,X,Y)

# ODE Solver
using Test

tol = 1.0e-6
sysslvr_l = LUSolver()
odeslvr = ThetaMethod(sysslvr_l,dt,0.5)
fesltn = solve(odeslvr,tfeop,t0,tF*10,xhs0)
for (t_n,xhs_n) in fesltn
  println("err H1(u): $(h1(u(t_n) - xhs_n[1]))")
  println("err L2(p): $(l2(p(t_n) - xhs_n[2]))")
end
