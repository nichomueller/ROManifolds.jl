using Gridap
using GridapEmbedded
using Test
using DrWatson
using Serialization

using ReducedOrderModels

# time marching
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

# geometry
R  = 0.5
L  = 0.5*R
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(-L,L)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.05
pmin = p1-t*R
pmax = p1+t*R
dp = pmax - pmin

n = 40
partition = (n,n)
bgmodel = TProductModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo3)
Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)

order = 2
degree = 2*order
Ω = Triangulation(bgmodel)
dΩ = Measure(Ω,degree)

order = 1
degree = 2*order

dΩ = Measure(Ω,degree)
dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)
Γ_in = EmbeddedBoundary(cutgeo)
nΓ_in = get_normal_vector(Γ_in)
dΓ_in = Measure(Γ_in,degree)

const γd = 10.0    # Nitsche coefficient
const h = dp[1]/n  # Mesh size according to the parameters of the background grid

ν_in(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
ν_in(μ,t) = x->ν_in(x,μ,t)
νμ_in(μ,t) = TransientParamFunction(ν_in,μ,t)

const ν_out = 1e-3

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

g0(x,μ,t) = 0
g0(μ,t) = x->g0(x,μ,t)
g0μt(μ,t) = TransientParamFunction(g0,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ_in,dΩ_out,dΓ_in) = ( ∫( νμ_in(μ,t)*∇(v)⋅∇(u) )dΩ_in + ∫( ν_out*∇(v)⋅∇(u) )dΩ_out
  + ∫( (γd/h)*v*u  - νμ_in(μ,t)*v*(nΓ_in⋅∇(u)) - νμ_in(μ,t)*(nΓ_in⋅∇(v))*u )dΓ_in )
mass(μ,t,uₜ,v,dΩ_in) = ∫(v*uₜ)dΩ_in
rhs(μ,t,v,dΩ_in,dΓ_in) = ∫(fμt(μ,t)*v)dΩ_in + ∫( (γd/h)*v*gμt(μ,t) - νμ_in(μ,t)*(nΓ_in⋅∇(v))*gμt(μ,t) )dΓ_in
res(μ,t,u,v,dΩ_in,dΩ_out,dΓ_in) = stiffness(μ,t,u,v,dΩ_in,dΩ_out,dΓ_in) + mass(μ,t,∂t(u),v,dΩ_in) - rhs(μ,t,v,dΩ_in,dΓ_in)

trians_res = (Ω_in,Ω_out,Γ_in)
trians_stiffness = trians_res
trians_mass = (Ω_in,)
domains = FEDomains(trians_res,(trians_stiffness,trians_mass))

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["boundary"])
trial = TransientTrialParamFESpace(test,g0μt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,trial,test,domains)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)

tol = fill(1e-4,4)
reduction = TTSVDReduction(tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,reduction;nparams_res=40,nparams_jac=20,nparams_djac=1)

fesnaps,festats = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
ronline = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,ronline)

x,festats = solution_snapshots(rbsolver,feop,ronline,uh0μ)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,ronline)

############################## WITH TPOD #####################################

bgmodel = CartesianDiscreteModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo3)
Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,PHYSICAL_IN)
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)

dΩ = Measure(Ω,degree)
dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)
Γ_in = EmbeddedBoundary(cutgeo)
nΓ_in = get_normal_vector(Γ_in)
dΓ_in = Measure(Γ_in,degree)

trians_res = (Ω_in,Ω_out,Γ_in)
trians_stiffness = trians_res
trians_mass = (Ω_in,)
domains = FEDomains(trians_res,(trians_stiffness,trians_mass))

test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["boundary"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,trial,test,domains)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

tol = 1e-4
energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
state_reduction = TransientReduction(tol,energy;nparams=100,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

fesnaps,festats = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
ronline = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,ronline)

x,festats = solution_snapshots(rbsolver,feop,ronline,uh0μ)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,ronline)

# plotting

# r = get_realization(fesnaps)
# S′ = flatten_snapshots(fesnaps)
# S1 = S′[:,:,1]
# r1 = r[1,:]
# U1 = trial(r1)
# plt_dir = datadir("plts")
# create_dir(plt_dir)
# using ReducedOrderModels.ParamDataStructures
# for i in 1:length(r1)
#   Ui = param_getindex(U1,i)
#   uhi = FEFunction(Ui,S1[:,i])
#   writevtk(Ω_in,joinpath(plt_dir,"sol_$i.vtu"),cellfields=["uh"=>uhi])
# end
