using Gridap
using Test
using DrWatson
using Serialization

using GridapEmbedded

using ReducedOrderModels

pranges = fill([1,10],3)
pspace = ParamSpace(pranges)

R  = 0.5
L  = 0.5*R
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(-L,L)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R
dp = pmax - pmin

n = 30
partition = (n,n)
bgmodel = TProductModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo3)
Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,ACTIVE_IN)
Ω_out = Triangulation(cutgeo,ACTIVE_OUT)

order = 1
degree = 2*order

dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)
Γ_in = EmbeddedBoundary(cutgeo)
nΓ_in = get_normal_vector(Γ_in)
dΓ_in = Measure(Γ_in,degree)

const γd = 10.0    # Nitsche coefficient
const h = dp[1]/n  # Mesh size according to the parameters of the background grid

ν_in(x,μ) = 1+exp(-x[1]/sum(μ))
ν_in(μ) = x->ν_in(x,μ)
νμ_in(μ) = ParamFunction(ν_in,μ)

const ν_out = 1e-6

f(x,μ) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]))
f(μ) = x->f(x,μ)
fμ(μ) = ParamFunction(f,μ)

g0(x,μ) = 0
g0(μ) = x->g0(x,μ)
g0μ(μ) = ParamFunction(g0,μ)

stiffness(μ,u,v,dΩ_in,dΩ_out,dΓ_in) = ( ∫( νμ_in(μ)*∇(v)⋅∇(u) )dΩ_in + ∫( ν_out*∇(v)⋅∇(u) )dΩ_out
  + ∫( (γd/h)*v*u  - v*(nΓ_in⋅∇(u)) - (nΓ_in⋅∇(v))*u )dΓ_in)
rhs(μ,v,dΓ_in) = ∫( (γd/h)*v*fμ(μ) - (nΓ_in⋅∇(v))*fμ(μ) )dΓ_in
res(μ,u,v,dΩ_in,dΩ_out,dΓ_in) = rhs(μ,v,dΓ_in) - stiffness(μ,u,v,dΩ_in,dΩ_out,dΓ_in)

reffe = ReferenceFE(lagrangian,Float64,order)

trians = (Ω_in.trian,Ω_out.trian,Γ_in)
test = TestFESpace(Ω,reffe,conformity=:H1;dirichlet_tags=["boundary"])
trial = ParamTrialFESpace(test,g0μ)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,trians,trians)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
energy(du,v) = ∫(v*du)dΩ_in + ∫(∇(v)⋅∇(du))dΩ_in
state_reduction = TTSVDReduction(tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

############################## WITH TPOD #####################################

bgmodel = CartesianDiscreteModel(pmin,pmax,partition)

cutgeo = cut(bgmodel,geo3)
Ω = Triangulation(bgmodel)
Ω_in = Triangulation(cutgeo,ACTIVE_IN)
Ω_out = Triangulation(cutgeo,ACTIVE_OUT)

dΩ = Measure(Ω,degree)
dΩ_in = Measure(Ω_in,degree)
dΩ_out = Measure(Ω_out,degree)
Γ_in = EmbeddedBoundary(cutgeo)
nΓ_in = get_normal_vector(Γ_in)
dΓ_in = Measure(Γ_in,degree)

trians = (Ω_in,Ω_out,Γ_in)
test = TestFESpace(Ω,reffe,conformity=:H1;dirichlet_tags=["boundary"])
trial = ParamTrialFESpace(test,g0μ)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,trians,trians)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
state_reduction = Reduction(tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = rb_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

#
using Gridap.FESpaces
using ReducedOrderModels.RBSteady

red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
op = get_algebraic_operator(feop)

jacs = jacobian_snapshots(rbsolver,op,fesnaps)
red = rbsolver.jacobian_reduction
basis = projection(red,s)
proj_basis = project(test,basis,trial)
indices,interp = empirical_interpolation(basis)
factor = lu(interp)
domain = integration_domain(indices)

form(u,v) = ∫(∇(v)⋅∇(u) )dΩ_in + ∫(∇(v)⋅∇(u))dΩ_out
AA = assemble_matrix(feop,form)
