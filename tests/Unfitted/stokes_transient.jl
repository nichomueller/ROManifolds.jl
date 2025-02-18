using Gridap
using GridapEmbedded
using Gridap.MultiField
using DrWatson
using Serialization

using ROM

θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 30*dt

pdomain = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 20
partition = (n,n)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = TProductDiscreteModel(pmin,pmax,partition)
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

u(x,μ,t) = VectorValue(μ[1]*x[1]*x[1],μ[2]*x[2])*exp(sin(μ[3]*t))
u(μ,t) = x->u(x,μ,t)
uμt(μ,t) = TransientParamFunction(u,μ,t)

dut(x,μ,t) = VectorValue(μ[1]*x[1]*x[1],μ[2]*x[2])*exp(sin(μ[3]*t))*cos(μ[3]*t)*μ[3]
dut(μ,t) = x->dut(x,μ,t)

p(x,μ,t) = (x[1]-x[2])*exp(cos(μ[3]*t))
p(μ,t) = x->p(x,μ,t)
pμt(μ,t) = TransientParamFunction(p,μ,t)

f(x,μ,t) = dut(x,μ,t) - Δ(u(μ,t))(x) + ∇(p(μ,t))(x)
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = (∇⋅u(μ,t))(x)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

g_0(x,μ,t) = VectorValue(0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

a(μ,t,(u,p),(v,q),dΩ,dΩ_out,dΓ) =
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p )dΩ +
  ∫( ∇(v)⊙∇(u) - q*p )dΩ_out +
  ∫( - v⋅(n_Γ⋅∇(u)) + (n_Γ⋅∇(v))⋅u + (p*n_Γ)⋅v + (q*n_Γ)⋅u )dΓ

res(μ,t,(u,p),(v,q),dΩ,dΩ_out,dΓ,dΓn) =
  ∫( v⋅fμt(μ,t) - q*gμt(μ,t) )dΩ +
  ∫( ∇(v)⊙∇(uμt(μ,t)) - pμt(μ,t)*q )dΩ_out +
  ∫( (n_Γ⋅∇(v))⋅uμt(μ,t) + (q*n_Γ)⋅uμt(μ,t) )dΓ +
  ∫( v⋅(n_Γn⋅∇(uμt(μ,t))) - (n_Γn⋅v)*pμt(μ,t) )dΓn

m(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ

u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

trian_res = (Ω,Ω_out,Γ,Γn)
trian_a = (Ω,Ω_out,Γ)
trian_m = (Ω,)
domains = FEDomains(trian_res,(trian_a,trian_m))

coupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TProductFESpace(Ωbg,reffe_u;conformity=:H1)
trial_u = TransientTrialParamFESpace(test_u)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TProductFESpace(Ωbg,reffe_p;conformity=:H1)
trial_p = TransientTrialParamFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = TransientParamLinearFEOperator((a,m),res,ptspace,trial,test,domains)

fesolver = ThetaMethod(LUSolver(),dt,θ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

tol = fill(1e-4,5)
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=20)

fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ)
save(fesnaps)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon,xh0μ)
save(x;label="online")
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon,Ω)
println(perf)

# r = get_realization(fesnaps)
# S′ = flatten(fesnaps)
# S1 = S′[1][:,:,1]
# r1 = r[1,:]
# U1 = trial_u(r1)
# plt_dir = datadir("plts")
# create_dir(plt_dir)
# for i in 1:length(r1)
#   Ui = param_getindex(U1,i)
#   uhi = FEFunction(Ui,S1[:,i])
#   writevtk(Ω,joinpath(plt_dir,"u_$i.vtu"),cellfields=["uh"=>uhi])
# end
# S2 = S′[2][:,:,1]
# for i in 1:length(r1)
#   Pi = trial_p
#   phi = FEFunction(Pi,S2[:,i])
#   writevtk(Ω,joinpath(plt_dir,"p_$i.vtu"),cellfields=["ph"=>phi])
# end
