module SteadyStokesTTSVD

using DrWatson
using Gridap
using ROM

import Gridap.MultiField: BlockMultiFieldStyle

include("ExamplesInterface.jl")

pranges = (1,10,-1,5,1,2)
pspace = ParamSpace(pranges)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
partition = (40,40)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = TProductModel(pmin,pmax,partition)
cutgeo = cut(bgmodel,geo2)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)
Γn = BoundaryTriangulation(Ωact,tags="boundary")

n_Γ = get_normal_vector(Γ)
n_Γn = get_normal_vector(Γn)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓn = Measure(Γn,degree)
dΩ_out = Measure(Ω_out,degree)

ν(x,μ) = μ[1]
ν(μ) = x->ν(x,μ)
νμ(μ) = ParamFunction(ν,μ)

u(x,μ) = VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)
u(μ) = x->u(x,μ)
uμ(μ) = ParamFunction(u,μ)

p(x,μ) = μ[1]*x[1]-x[2]
p(μ) = x->p(x,μ)
pμ(μ) = ParamFunction(p,μ)

f(x,μ) = - ν(x,μ)*Δ(u(μ))(x) + ∇(p(μ))(x)
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g(x,μ) = (∇⋅u(μ))(x)
g(μ) = x->g(x,μ)
gμ(μ) = ParamFunction(g,μ)

g_0(x,μ) = VectorValue(0.0,0.0)
g_0(μ) = x->g_0(x,μ)
gμ_0(μ) = ParamFunction(g_0,μ)

a(μ,(u,p),(v,q),dΩ,dΩ_out,dΓ) = (
  ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ +
  ∫( ∇(v)⊙∇(u) - p*q )dΩ_out +
  ∫( - v⋅(n_Γ⋅∇(u))*νμ(μ) + (n_Γ⋅∇(v))⋅u*νμ(μ) + (p*n_Γ)⋅v + (q*n_Γ)⋅u )dΓ
)

l(μ,(u,p),(v,q),dΩ,dΩ_out,dΓ,dΓn) = (
  ∫( v⋅fμ(μ) - q*gμ(μ) )dΩ +
  ∫( ∇(v)⊙∇(uμ(μ)) - pμ(μ)*q )dΩ_out +
  ∫( (n_Γ⋅∇(v))⋅uμ(μ)*νμ(μ) + (q*n_Γ)⋅uμ(μ) )dΓ +
  ∫( v⋅(n_Γn⋅∇(uμ(μ)))*νμ(μ) - (n_Γn⋅v)*pμ(μ) )dΓn
)

trian_res = (Ω,Ω_out,Γ,Γn)
trian_jac = (Ω,Ω_out,Γ)
domains = FEDomains(trian_res,trian_jac)

coupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩbg + ∫(dp*∂₂(v))dΩbg
energy((du,dp),(v,q)) = ∫(du⋅v)dΩbg + ∫(∇(v)⊙∇(du))dΩbg + ∫(dp*q)dΩbg

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TProductFESpace(Ωbg,reffe_u;conformity=:H1)
trial_u = ParamTrialFESpace(test_u)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TProductFESpace(Ωbg,reffe_p;conformity=:H1)
trial_p = TrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamFEOperator(l,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = fill(1e-4,4)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=50)

dir = datadir("stokes_ttsvd")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
run_test(dir,rbsolver,feop,tols)

end
