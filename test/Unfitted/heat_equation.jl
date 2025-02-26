using Gridap
using Test
using DrWatson
using Serialization

using GridapEmbedded

using ROManifolds

# time marching
θ = 1
dt = 0.01
t0 = 0.0
tf = 0.2

# parametric space
pdomain = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

R = 0.5
L = 0.8*(2*R)
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(L,0.0)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R
dp = pmax - pmin

n = 20
partition = (n,n)
bgmodel = TProductDiscreteModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo)
Ωbg = Triangulation(bgmodel)
Ω = Triangulation(cutgeo,PHYSICAL)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩ_out = Measure(Ω_out,degree)
dΓ = Measure(Γ,degree)

nΓ = get_normal_vector(Γ)

ν(x,μ,t) = μ[1]
ν(μ,t) = x->ν(x,μ,t)
νμt(μ,t) = TransientParamFunction(ν,μ,t)

f(x,μ,t) = abs(sin(t/μ[2]))
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = exp(-x[1]/μ[2])
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

# non - symmetric formulation

a(μ,t,u,v,dΩ,dΩ_out,dΓ) = ∫( νμt(μ,t)*∇(v)⋅∇(u) )dΩ + ∫( ∇(v)⋅∇(u) )dΩ_out - ∫( νμt(μ,t)*v*(nΓ⋅∇(u)) - νμt(μ,t)*(nΓ⋅∇(v))*u )dΓ
m(μ,t,dut,v,dΩ) = ∫( dut*v )dΩ
b(μ,t,u,v,dΩ,dΩ_out,dΓ) = ∫( ∂t(u)*v )dΩ + ∫( νμt(μ,t)*∇(v)⋅∇(u) )dΩ + ∫( ∇(v)⋅∇(gμt(μ,t)) )dΩ_out + ∫( νμt(μ,t)*(nΓ⋅∇(v))*gμt(μ,t) )dΓ - ∫( v*fμt(μ,t) )dΩ

domains = FEDomains((Ω,Ω_out,Γ),((Ω,Ω_out,Γ),(Ω,)))

reffe = ReferenceFE(lagrangian,Float64,order)

test = TProductFESpace(Ωbg,reffe,conformity=:H1)
trial = TransientTrialParamFESpace(test)
feop = TransientParamLinearFEOperator((a,m),b,ptspace,trial,test,domains)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

tol = 1e-4
energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
state_reduction = TTSVDReduction(tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=30,nparams_jac=10)

fesnaps,festats = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)

x,festats = solution_snapshots(rbsolver,feop,μon,uh0μ)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon,Ω)

# plotting
using ROManifolds.ParamDataStructures
r = μon
r1 = r[1,:]
S1 = get_all_data(x)[:,1:10:end]
U1 = trial(r1)
plt_dir = datadir("plts")
create_dir(plt_dir)
for i in 1:length(r1)
  Ui = param_getindex(U1,i)
  uhi = FEFunction(Ui,S1[:,i])
  writevtk(Ω,joinpath(plt_dir,"sol_$i.vtu"),cellfields=["uh"=>uhi])
end

############################## WITH TPOD #####################################

bgmodel = CartesianDiscreteModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo)
Ωbg = Triangulation(bgmodel)
Ω = Triangulation(cutgeo,PHYSICAL)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩ_out = Measure(Ω_out,degree)
dΓ = Measure(Γ,degree)

nΓ = get_normal_vector(Γ)

domains = FEDomains((Ω,Ω_out,Γ),((Ω,Ω_out,Γ),(Ω,)))

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ωbg,reffe;conformity=:H1)
trial = TransientTrialParamFESpace(test)
feop = TransientParamLinearFEOperator((a,m),b,ptspace,trial,test,domains)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

tol = 1e-4
energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
state_reduction = TransientReduction(tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=30,nparams_jac=10)

fesnaps′,festats′ = solution_snapshots(rbsolver,feop,uh0μ)
rbop′ = reduced_operator(rbsolver,feop,fesnaps′)

ronline = realization(feop;nparams=1)
x̂′,rbstats′ = solve(rbsolver,rbop′,ronline,uh0μ)

x′,festats′ = solution_snapshots(rbsolver,feop,ronline,uh0μ)
perf′ = eval_performance(rbsolver,feop,rbop′,x′,x̂′,festats′,rbstats′,ronline,Ω)

# plotting
xrb = Snapshots(inv_project(rbop′.trial(ronline),x̂′),get_dof_map(feop),ronline)

using ROManifolds.ParamDataStructures
r = ronline
r1 = r[1,:]
S1 = get_all_data(x′)
Ŝ1 = get_all_data(xrb)
U1 = trial(r1)
plt_dir = datadir("plts")
create_dir(plt_dir)
for i in 1:length(r1)
  Ui = param_getindex(U1,i)
  uhi = FEFunction(Ui,S1[:,i])
  ûhi = FEFunction(Ui,Ŝ1[:,i])
  ehi = uhi - ûhi
  writevtk(Ω,joinpath(plt_dir,"sol_$i.vtu"),cellfields=["uh"=>uhi,"ûh"=>ûhi,"eh"=>ehi])
end

# GRIDAP

μ = realization(ptspace).params.params[1]

m(t,du,v) = ∫( du*v )dΩ
a(t,du,v) = ∫( ν(μ,t)*∇(v)⋅∇(du) )dΩ + ∫( ∇(v)⋅∇(du) )dΩ_out - ∫( ν(μ,t)*v*(nΓ⋅∇(du)) - ν(μ,t)*(nΓ⋅∇(v))*du )dΓ
l(t,v) = ∫( ∇(v)⋅∇(g(μ,t)) )dΩ_out + ∫( v*f(μ,t) )dΩ - ∫(ν(μ,t)*(nΓ⋅∇(v))*g(μ,t) )dΓ

jac(t,u,du,v) = a(t,du,v)
jac_t(t,u,du,v) = m(t,du,v)
res(t,u,v) = m(t,∂t(u),v) + a(t,u,v) - l(t,v)

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
trial = TransientTrialFESpace(test)
feop = TransientFEOperator(res,(jac,jac_t),trial,test)

uh0 = interpolate_everywhere(u0(μ),trial(t0))
uht = solve(fesolver,feop,t0,tf,uh0)

S = Vector{Float64}[]
for (tn, uhn) in uht
  # writevtk(Ω,joinpath(plt_dir,"solgridap_$tn.vtu"),cellfields=["uh"=>uhn])
  push!(S,copy(get_free_dof_values(uhn)))
end

feop_lin = TransientLinearFEOperator((a,m),l,trial,test)

fesolver_lin = ThetaMethod(LUSolver(),dt,θ)
uh0 = interpolate_everywhere(u0(μ),trial(t0))
uht_lin = solve(fesolver_lin,feop_lin,t0,tf,uh0)


S_lin = Vector{Float64}[]
for (tn, uhn) in uht_lin
  push!(S_lin,copy(get_free_dof_values(uhn)))
end
