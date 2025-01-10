module ElasticitySteady

using ROM
using Gridap
using DrWatson

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

include("ExamplesInterface.jl")

θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 40*dt

pranges = (1e2,5*1e2,0.25,1.25,1e-3,1e-2,1e-3,1e-2)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,0.5,0,0.25)
partition = (20,10,5)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet0",[1,3,5,7,13,15,17,19,25])
add_tag_from_tags!(labels,"dirichlet",[2,4,6,8,14,16,18,20,26])

Ω = Triangulation(model)

λ(μ) = μ[1]
p(μ) = μ[2]

# Deformation Gradient
F(∇u) = one(∇u) + ∇u'
J(F) = sqrt(det(C(F)))
dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )

# Right Cauchy-green deformation tensor
C(F) = (F')⋅F

# Constitutive law (Neo hookean)

function S(∇u,μ,t)
  Cinv = inv(C(F(∇u)))
  p(μ)*(one(∇u)-Cinv) + λ(μ)*log(J(F(∇u)))*Cinv
end
S(μ,t) = ∇u -> S(∇u,μ,t)
Sμt(μ,t) = TransientParamFunction(S,μ,t)

function dS(∇du,∇u,μ,t)
  Cinv = inv(C(F(∇u)))
  _dE = dE(∇du,∇u)
  λ(μ)*(Cinv⊙_dE)*Cinv + 2*(p(μ)-λ(μ)*log(J(F(∇u))))*Cinv⋅_dE⋅(Cinv')
end
dS(μ,t) = (∇du,∇u) -> dS(∇du,∇u,μ,t)
dSμt(μ,t) = TransientParamFunction(dS,μ,t)

σ(∇u,μ,t) = (1.0/J(F(∇u)))*F(∇u)⋅S(μ,∇u)⋅(F(∇u))'
σ(μ,t) = ∇u -> σ(∇u,μ,t)
σμt(μ,t) = TransientParamFunction(σ,μ,t)

g(x,μ,t) = VectorValue(0.0,μ[3],μ[4])
g(μ,t) = x -> g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

g0(x,μ,t) = VectorValue(0.0,0.0,0.0)
g0(μ,t) = x -> g0(x,μ,t)
g0μt(μ,t) = TransientParamFunction(g0,μ,t)

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

order = 1
reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
trial = TransientTrialParamFESpace(test,[g0μt,gμt])

degree = 2*order
dΩ = Measure(Ω,degree)

res(μ,t,u,v,dΩ) = ∫(v⋅∂t(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
stiffness(μ,t,u,v,dΩ) = ∫(0*v⋅u)dΩ

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)
domains_lin = FEDomains(trian_res,(trian_jac,trian_jac_t))

res_nlin(μ,t,u,v,dΩ) = ∫( (dE∘(∇(v),∇(u))) ⊙ (Sμt(μ,t)∘∇(u)) )dΩ
jac_nlin(μ,t,u,du,v,dΩ) = (
  ∫( (dE∘(∇(v),∇(u))) ⊙ (dSμt(μ,t)∘(∇(du),∇(u))) )dΩ +
  ∫( ∇(v) ⊙ ( (Sμt(μ,t)∘∇(u))⋅∇(du) ) )dΩ
  )

trian_res_nlin = (Ω,)
trian_jac_nlin = (Ω,)
domains_nlin = FEDomains(trian_res_nlin,(trian_jac_nlin,))

feop_lin = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,domains_lin;constant_forms=(true,true))
feop_nlin = TransientParamFEOperator(res_nlin,jac_nlin,ptspace,
  trial,test,domains_nlin)
feop = LinearNonlinearTransientParamFEOperator(feop_lin,feop_nlin)

fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

tol = 1e-5
energy(du,v) = ∫(v⋅du)dΩ + ∫(∇(v)⊙∇(du))dΩ
state_reduction = TransientReduction(tol,energy;nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=20,nparams_jac=20,nparams_djac=1)

dir = datadir("transient_hyper_elasticity_pod")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
ExamplesInterface.run_test(dir,rbsolver,feop,tols,uh0μ)

end

r = realization(ptspace)
μ = get_params(r).params[1]

_S(∇u,t) = S(∇u,μ,t)
_S(t) = ∇u -> _S(∇u,t)
_dS(∇du,∇u,t) = dS(∇du,∇u,μ,t)
_dS(t) = (∇du,∇u) -> _dS(∇du,∇u,t)
_g(x,t) = g(x,μ,t)
_g(t) = x -> _g(x,t)
_g0(x,t) = g0(x,μ,t)
_g0(t) = x -> _g0(x,t)
_σ(∇u,t) = σ(∇u,μ,t)
_σ(t) = ∇u -> _σ(∇u,t)

_res(t,u,v) = ∫(v⋅∂t(u))dΩ + ∫( (dE∘(∇(v),∇(u))) ⊙ (_S(t)∘∇(u)) )dΩ
_jac_t(t,u,du,v) = ∫(v⋅du)dΩ
_jac(t,u,du,v) = ∫( (dE∘(∇(v),∇(u))) ⊙ (_dS(t)∘(∇(du),∇(u))) + ∇(v) ⊙ ( (_S(t)∘∇(u))⋅∇(du) ) )dΩ

U = TransientTrialFESpace(test,[_g0,_g])
_feop = TransientFEOperator(_res,(_jac,_jac_t),U,test)

_uh0 = interpolate_everywhere(x->VectorValue(0,0,0),U(t0))

uh = solve(fesolver,_feop,t0,tf,_uh0)
sol = Vector{Float64}[]
for (tn,uhn) in uh
  push!(sol,copy(get_free_dof_values(uhn)))
end

using Gridap.ODEs

odesltn = uh.odesltn
odeslvr, odeop = odesltn.odeslvr, odesltn.odeop
t0, us0 = odesltn.t0, odesltn.us0

# Allocate cache
odecache = allocate_odecache(odeslvr, odeop, t0, us0)

# Starting procedure
state0, odecache = ode_start(
  odeslvr, odeop,
  t0, us0,
  odecache
)

# Marching procedure
first_state = copy.(state0)
stateF = copy.(state0)

# Unpack inputs
w0 = state0[1]
odeslvrcache, odeopcache = odecache
uθ, sysslvrcache = odeslvrcache

# Unpack solver
odeslvr = fesolver
sysslvr = odeslvr.sysslvr
dt, θ = odeslvr.dt, odeslvr.θ

# Define scheme
x = stateF[1]
dtθ = θ * dt
tx = t0 + dtθ
function _usx(x)
  copy!(uθ, w0)
  axpy!(dtθ, x, uθ)
  (uθ, x)
end
ws = (dtθ, 1)

# Update ODE operator cache
update_odeopcache!(odeopcache, odeop, tx)

# Create and solve stage operator
stageop = NonlinearStageOperator(
  odeop, odeopcache,
  tx, _usx, ws
)

sysslvrcache = solve!(x, sysslvr, stageop, sysslvrcache)

# Update state
tF = t0 + dt
stateF = ODEs._udate_theta!(stateF, state0, dt, x)

state0 = copy.(stateF)
stateF = copy.(first_state)
t0 = tF

odeslvrcache = (uθ, sysslvrcache)
odecache = (odeslvrcache, odeopcache)
