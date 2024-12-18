using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 20
partition = (n,n)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = TProductModel(pmin,pmax,partition)
cutgeo = cut(bgmodel,geo2)
cutgeo_facets = cut_facets(bgmodel,geo2)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)
Γi = SkeletonTriangulation(cutgeo_facets)
Γn = BoundaryTriangulation(Ωact,tags="boundary")

n_Γ = get_normal_vector(Γ)
n_Γg = get_normal_vector(Γg)
n_Γi = get_normal_vector(Γi)
n_Γn = get_normal_vector(Γn)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)
dΓi = Measure(Γi,degree)
dΓn = Measure(Γn,degree)
dΩ_out = Measure(Ω_out,degree)

u(x,μ) = VectorValue(μ[1]*x[1]*x[1],μ[2]*x[2])
u(μ) = x->u(x,μ)
uμ(μ) = ParamFunction(u,μ)

p(x,μ) = μ[3]*x[1]-x[2]
p(μ) = x->p(x,μ)
pμ(μ) = ParamFunction(p,μ)

f(x,μ) = - Δ(u(μ))(x) + ∇(p(μ))(x)
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g(x,μ) = (∇⋅u(μ))(x)
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)

g_0(x,μ) = VectorValue(0.0,0.0)
g_0(μ) = x->g_0(x,μ)
gμ_0(μ) = ParamFunction(g_0,μ)

const ν0 = 1e-3
const h = (pmax - pmin)[1]/n

β0 = 0.25
β1 = 0.2
β2 = 0.1
β3 = 0.05
γ = 10.0

a_Ω(u,v) = ∇(u)⊙∇(v)
b_Ω(v,p) = - (∇⋅v)*p
c_Γi(p,q) = (β0*h)*jump(p)*jump(q)
c_Ω(p,q) = (β1*h^2)*∇(p)⋅∇(q)
e_Ωout((u,p),(v,q)) = ν0*(∇(v)⊙∇(u) + q*p)
a_Γ(u,v) = - (n_Γ⋅∇(u))⋅v - u⋅(n_Γ⋅∇(v)) + (γ/h)*u⋅v
b_Γ(v,p) = (n_Γ⋅v)*p
i_Γg(u,v) = (β2*h)*jump(n_Γg⋅∇(u))⋅jump(n_Γg⋅∇(v))
j_Γg(p,q) = (β3*h^3)*jump(n_Γg⋅∇(p))*jump(n_Γg⋅∇(q)) + c_Γi(p,q)
ϕ_Ω(μ,q) = (β1*h^2)*∇(q)⋅fμ(μ)

a(μ,(u,p),(v,q),dΩ,dΩ_out,dΓ,dΓg) =
  (∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)-c_Ω(p,q) ) * dΩ +
  ∫( e_Ωout((u,p),(v,q)) )dΩ_out +
  ∫( a_Γ(u,v)+b_Γ(u,q)+b_Γ(v,p) ) * dΓ +
  ∫( i_Γg(u,v) - j_Γg(p,q) ) * dΓg)

l(μ,(u,p),(v,q),dΩ,dΓ,dΓn) =
  ∫( v⋅fμ(μ) - ϕ_Ω(μ,q) - q*gμ(μ) ) * dΩ +
  ∫( uμ(μ)⊙( (γ/h)*v - n_Γ⋅∇(v) + q*n_Γ ) ) * dΓ +
  ∫( v⋅(n_Γn⋅∇(u)) - (n_Γn⋅v)*p ) * dΓn

trian_res = (Ω,Γ,Γn)
trian_jac = (Ω,Ω_out,Γ,Γg)
domains = FEDomains(trian_res,trian_jac)

coupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩbg + ∫(dp*∂₂(v))dΩbg
energy((du,dp),(v,q)) = ∫(du⋅v)dΩbg + ∫(∇(v)⊙∇(du))dΩbg + ∫(dp*q)dΩbg

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TProductFESpace(Ωbg,reffe_u;conformity=:H1,dirichlet_tags="boundary")
trial_u = ParamTrialFESpace(test_u,uμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TProductFESpace(Ωbg,reffe_p;conformity=:H1,constraint=:zeromean)
trial_p = TrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = fill(1e-4,4)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=50)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10,random=true)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

# u1 = flatten_snapshots(x[1])[:,1]
# p1 = flatten_snapshots(x[2])[:,1]
# r1 = get_realization(x[1])[1]
# U1 = param_getindex(trial_u(r1),1)
# uh = FEFunction(U1,u1)
# ph = FEFunction(trial_p,p1)
# writevtk(Ω,datadir("plts/sol"),cellfields=["uh"=>uh,"ph"=>ph])

# xrb = Snapshots(inv_project(get_trial(rbop)(μon),x̂),get_dof_map(feop),μon)
# u1 = flatten_snapshots(xrb[1])[:,1]
# p1 = flatten_snapshots(xrb[2])[:,1]
# uh = FEFunction(U1,u1)
# ph = FEFunction(trial_p,p1)
# writevtk(Ω,datadir("plts/sol_approx"),cellfields=["uh"=>uh,"ph"=>ph])

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 20
partition = (n,n)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
cutgeo = cut(bgmodel,geo2)
cutgeo_facets = cut_facets(bgmodel,geo2)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)
Γi = SkeletonTriangulation(cutgeo_facets)
Γn = BoundaryTriangulation(Ωact,tags="boundary")

n_Γ = get_normal_vector(Γ)
n_Γg = get_normal_vector(Γg)
n_Γi = get_normal_vector(Γi)
n_Γn = get_normal_vector(Γn)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)
dΓi = Measure(Γi,degree)
dΓn = Measure(Γn,degree)
dΩ_out = Measure(Ω_out,degree)

# μ = get_realization(fesnaps).params[1]
# uu(x) = VectorValue(μ[1]*x[1]*x[1],μ[2]*x[2])
# pp(x) = μ[3]*x[1]-x[2]
uu(x) = VectorValue(x[1]*x[1],x[2])
pp(x) = x[1]-x[2]
ff(x) = - Δ(uu)(x) + ∇(pp)(x)
gg(x) = (∇⋅uu)(x)

ϕϕ_Ω(q) = (β1*h^2)*∇(q)⋅ff

aa((u,p),(v,q)) =
  (∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)-c_Ω(p,q) ) * dΩ +
  ∫( e_Ωout((u,p),(v,q)) )dΩ_out +
  ∫( a_Γ(u,v)+b_Γ(u,q)+b_Γ(v,p) ) * dΓ +
  ∫( i_Γg(u,v) - j_Γg(p,q) ) * dΓg)

ll((v,q)) =
  ∫( v⋅ff - ϕϕ_Ω(q) - q*gg ) * dΩ +
  ∫( uu⊙( (γ/h)*v - n_Γ⋅∇(v) + q*n_Γ ) ) * dΓ +
  ∫( v⋅(n_Γn⋅∇(uu)) - (n_Γn⋅v)*pp ) * dΓn

op = AffineFEOperator(aa,ll,test,test)
uh,ph = solve(op)

l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))

eu_h1 = h1(uu - uh)
ep_l2 = l2(pp - ph)

aaa((u,p),(v,q)) =
  (∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)-c_Ω(p,q) ) * dΩ +
  ∫( a_Γ(u,v)+b_Γ(u,q)+b_Γ(v,p) ) * dΓ +
  ∫( i_Γg(u,v) - j_Γg(p,q) ) * dΓg)

lll((v,q)) =
  ∫( v⋅ff - ϕϕ_Ω(q) - q*gg ) * dΩ +
  ∫( uu⊙( (γ/h)*v - n_Γ⋅∇(v) + q*n_Γ ) ) * dΓ +
  ∫( v⋅(n_Γn⋅∇(uu)) - (n_Γn⋅v)*pp ) * dΓn

V = FESpace(Ωact,reffe_u;conformity=:H1,dirichlet_tags="boundary")
Q = FESpace(Ωact,reffe_p;conformity=:H1,constraint=:zeromean)
Y = MultiFieldParamFESpace([V,Q])
X = Y

oop = AffineFEOperator(aaa,lll,Y,X)
uuh,pph = solve(oop)

writevtk(Ω,datadir("plts/err_cutfem"),cellfields=["euh"=>uh-uuh,"eph"=>ph-pph])

eeu_h1 = h1(uu - uuh)
eep_l2 = l2(pp - pph)
