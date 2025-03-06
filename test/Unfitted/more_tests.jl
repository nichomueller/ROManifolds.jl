using Gridap
using GridapEmbedded
using ROManifolds

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

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
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)
Ωact = Triangulation(cutgeo,ACTIVE)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωout,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)

nΓ = get_normal_vector(Γ)

ν(μ) = x->sum(μ)
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

g(μ) = x->μ[3]*sum(x)
gμ(μ) = ParamFunction(g,μ)

# non-symmetric formulation

a(μ,u,v,dΩ,dΩout,dΓ) = ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( νμ(μ)*∇(v)⋅∇(u) )dΩout - ∫( νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
b(μ,u,v,dΩ,dΩout,dΓ) = a(μ,u,v,dΩ,dΓ) - ∫( νμ(μ)*∇(v)⋅∇(gμ(μ)) )dΩout - ∫( v*fμ(μ) )dΩ - ∫( νμ(μ)*(nΓ⋅∇(v))*gμ(μ) )dΓ
domains = FEDomains((Ω,Ωout,Γ),(Ω,Ωout,Γ))

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(bgmodel,geo)

reffe = ReferenceFE(lagrangian,Float64,order)
test = TProductFESpace(Ωbg,reffe;conformity=:H1)
trial = ParamTrialFESpace(test,gμ)
feop = LinearParamFEOperator(b,a,pspace,trial,test,domains)

tol = 1e-4
energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
state_reduction = TTSVDReduction(tol,energy;nparams=100)
fesolver = LUSolver()
rbsolver = RBSolver(fesolver,state_reduction)

# offline
fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

# online
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)

# test
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon;internal_nodes=false)

# last attempt
Ωincut = Ω.a
dΩincut = Measure(Ωincut,degree)
a(μ,u,v,dΩ,dΩincut,dΓ) = ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ + ∫( νμ(μ)*∇(v)⋅∇(u) )dΩincut - ∫( νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
b(μ,u,v,dΩ,dΓ) = a(μ,u,v,dΩ,dΓ) - ∫( v*fμ(μ) )dΩ - ∫( νμ(μ)*(nΓ⋅∇(v))*gμ(μ) )dΓ
domains = FEDomains((Ω,Γ),(Ω,Ωincut,Γ))
bgcell_to_inoutcut = compute_bgcell_to_inoutcut(bgmodel,geo)
test = TProductFESpace(Ωact,Ωbg,bgcell_to_inoutcut,reffe;conformity=:H1)
trial = ParamTrialFESpace(test,gμ)
feop = LinearParamFEOperator(b,a,pspace,trial,test,domains)
fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon;internal_nodes=false)
#

rbsnaps = RBSteady.to_snapshots(rbop.trial,x̂,μon)
i′ = get_internal_dof_map(feop)
rbsnaps′ = change_dof_map(rbsnaps,i′)
fesnaps′ = change_dof_map(x,i′)

# ttsvd
bgmodel′ = bgmodel.model
Ωbg′ = Ωbg.trian
dΩbg′ = dΩbg.measure
test′ = TestFESpace(Ωact,reffe;conformity=:H1)
trial′ = ParamTrialFESpace(test′,gμ)
feop′ = LinearParamFEOperator(b,a,pspace,trial′,test′,domains)
state_reduction′ = PODReduction(tol,energy;nparams=100)
rbsolver′ = RBSolver(fesolver,state_reduction′)
fesnaps′, = solution_snapshots(rbsolver′,feop′)
rbop′ = reduced_operator(rbsolver′,feop′,fesnaps′)
x̂′,rbstats′ = solve(rbsolver′,rbop′,μon)
x′,festats′ = solution_snapshots(rbsolver′,feop′,μon)
perf′ = eval_performance(rbsolver′,feop′,rbop′,x′,x̂′,festats′,rbstats′,μon)

Φ = get_basis(rbop.test.subspace)
Φ′ = get_basis(rbop′.test.subspace)

using Gridap.FESpaces
using Gridap.Algebra
r = μon
op = rbop
y = zero_free_values(trial(r))
ŷ = zero_free_values(rbop.trial(r))
rbcache = allocate_rbcache(op,r,y)
Â = jacobian(op,r,y,rbcache)
b̂ = residual(op,r,y,rbcache)
x̂1 = param_getindex(Â,1) \ param_getindex(b̂,1)

errs = x[:,:,1] - rbsnaps[:,:,1]

heatmap(z=x[:,:,1] - rbsnaps[:,:,1])

using Plots
dx = collect(0:1/40:1)
heatmap(dx,dx,errs)

op′ = rbop′
y′ = zero_free_values(trial′(r))
ŷ′ = zero_free_values(rbop′.trial(r))
rbcache′ = allocate_rbcache(op′,r,y′)
Â′ = jacobian(op′,r,y′,rbcache′)
b̂′ = residual(op′,r,y′,rbcache′)
x̂1′ = param_getindex(Â′,1) \ param_getindex(b̂′,1)

STOP
# using ROManifolds.RBSteady
# rbsnaps = RBSteady.to_snapshots(rbop.trial,x̂,μon)

# v = x[:,:,1]
# x̂ = project(rbop.test.subspace,v)

# using DrWatson
# using ROManifolds.ParamDataStructures
# r1 = get_realization(x)[1]
# S1 = get_param_data(x)[1]
# Ŝ1 = get_param_data(rbsnaps)[1]
# plt_dir = datadir("plts")
# create_dir(plt_dir)
# uh1 = FEFunction(param_getindex(trial(r1),1),S1)
# ûh1 = FEFunction(param_getindex(trial(r1),1),Ŝ1)
# writevtk(Ω,joinpath(plt_dir,"sol.vtu"),cellfields=["uhapp"=>ûh1,"uh"=>uh1,"eh"=>uh1-ûh1])
# writevtk(Ωact,joinpath(plt_dir,"solact.vtu"),cellfields=["uhapp"=>ûh1,"uh"=>uh1,"eh"=>uh1-ûh1])

A = copy(fesnaps)
red_style = state_reduction.red_style
cores, = ttsvd(red_style,A)
Φ = cores2basis(cores...)

v = vec(copy(x[:,:,1]))
maximum(abs.(v - Φ*Φ'*v))

X = assemble_matrix(feop,energy)
Xmat = kron(X)
cores, = ttsvd(red_style,A,X)
Φ = cores2basis(cores...)
maximum(abs.(v - Φ*Φ'*Xmat*v))

using ROManifolds
using Gridap
using DrWatson

# geometry
Ω = (0,1,0,1)
parts = (10,10)
Ωₕ = CartesianDiscreteModel(Ω,parts)
τₕ = Triangulation(Ωₕ)

# temporal grid
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 10*dt
tdomain = t0:dt:tf

# parametric quantities
pdomain = (1,5,1,5)
D  = TransientParamSpace(pdomain,tdomain)
u(μ,t) = x -> t*(μ[1]*x[1]^2 + μ[2]*x[2]^2)
uₚₜ(μ,t) = parameterize(u,μ,t)
f(μ,t) = x -> -Δ(u(μ,t))(x)
fₚₜ(μ,t) = parameterize(f,μ,t)

# numerical integration
order = 1
dΩₕ = Measure(τₕ,2order)

# weak form
a(μ,t,du,v,dΩₕ) = ∫(∇(v)⋅∇(du))dΩₕ
m(μ,t,du,v,dΩₕ) = ∫(v*du)dΩₕ
r(μ,t,u,v,dΩₕ) = m(μ,t,∂t(u),v,dΩₕ) + a(μ,t,u,v,dΩₕ) - ∫(fₚₜ(μ,t)*v)dΩₕ

# triangulation information
τₕ_a = (τₕ,)
τₕ_m = (τₕ,)
τₕ_r = (τₕ,)
domains = FEDomains(τₕ_r,(τₕ_a,τₕ_m))

# FE interpolation
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(Ωₕ,reffe;dirichlet_tags="boundary")
U = TransientTrialParamFESpace(V,uₚₜ)
feop = TransientParamLinearFEOperator((a,m),r,D,U,V,domains)

# initial condition
u₀(μ) = x -> 0.0
u₀ₚ(μ) = parameterize(u₀,μ)
uh₀ₚ(μ) = interpolate_everywhere(u₀ₚ(μ),U(μ,t0))

# FE solver
slvr = ThetaMethod(LUSolver(),dt,θ)

# RB solver
tol = 1e-4
inner_prod(u,v) = ∫(∇(v)⋅∇(u))dΩₕ
red_sol = TransientReduction(tol,inner_prod;nparams=20)
rbslvr = RBSolver(slvr,red_sol;nparams_jac=1,nparams_res=20)

dir = datadir("heat_equation")
create_dir(dir)

rbop = try # try loading offline quantities
    load_operator(dir,feop)
catch # offline phase
    reduced_operator(dir,rbslvr,feop,uh₀ₚ)
end

# online phase
μₒₙ = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbslvr,rbop,μₒₙ,uh₀ₚ)

# post process
x,stats = solution_snapshots(slvr,feop,μₒₙ,uh₀ₚ)
eval_performance(rbslvr,feop,rbop,x,x̂,stats,rbstats)
