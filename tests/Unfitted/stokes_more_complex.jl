using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ROM

pranges = (1,10,-1,5,1,2)
pspace = ParamSpace(pranges)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 20
partition = (n,n)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = TProductModel(pmin,pmax,partition)
labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])

cutgeo = cut(bgmodel,geo2)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ωact_out = Triangulation(cutgeo,ACTIVE_OUT)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)

n_Γ = get_normal_vector(Γ)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΩ_out = Measure(Ω_out,degree)

ν(x,μ) = μ[1]
ν(μ) = x->ν(x,μ)
νμ(μ) = ParamFunction(ν,μ)

g(x,μ) = VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)

f(x,μ) = VectorValue(0.0,0.0)
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g_0(x,μ) = VectorValue(0.0,0.0)
g_0(μ) = x->g_0(x,μ)
gμ_0(μ) = ParamFunction(g_0,μ)

a(μ,(u,p),(v,q)) = (
  ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ +
  ∫( ∇(v)⊙∇(u) )dΩ_out +
  ∫( - v⋅(n_Γ⋅∇(u))*νμ(μ) + (n_Γ⋅∇(v))⋅u*νμ(μ) + (p*n_Γ)⋅v + (q*n_Γ)⋅u )dΓ
)

l(μ,(u,p),(v,q)) = (
  ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ +
  ∫( ∇(v)⊙∇(gμ_0(μ)) )dΩ_out +
  ∫( (n_Γ⋅∇(v))⋅gμ_0(μ)*νμ(μ) + (q*n_Γ)⋅gμ_0(μ) )dΓ
)

trian_res = (Ω,Ω_out,Γ)
trian_jac = (Ω,Ω_out,Γ)
domains = FEDomains(trian_res,trian_jac)

coupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩbg + ∫(dp*∂₂(v))dΩbg
energy((du,dp),(v,q)) = ∫(du⋅v)dΩbg + ∫(∇(v)⊙∇(du))dΩbg + ∫(dp*q)dΩbg

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
# test_u = TProductFESpace(Ωbg,reffe_u;conformity=:H1)
test_u = FESpace(Ωbg.trian,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
# test_p = TProductFESpace(Ωact,Ωbg,reffe_p;conformity=:H1)
test_p = FESpace(Ωact,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test)

fesolver = LinearFESolver(LUSolver())

tol = fill(1e-4,4)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=50)

x, = solution_snapshots(rbsolver,feop;nparams=1)
u1 = flatten_snapshots(x[1])[:,1]
p1 = flatten_snapshots(x[2])[:,1]
r1 = get_realization(x[1])[1]
U1 = param_getindex(trial_u(r1),1)
P1 = trial_p(nothing)
uh = FEFunction(U1,u1)
ph = FEFunction(P1,p1)
writevtk(Ω,datadir("plts/sol"),cellfields=["uh"=>uh,"ph"=>ph])
writevtk(Ωbg.trian,datadir("plts/sol_bg"),cellfields=["uh"=>uh,"ph"=>ph])
