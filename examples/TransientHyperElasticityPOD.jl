module ElasticitySteady

using ROM
using Gridap
using DrWatson

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 40*dt

pranges = (1e10,9*1e10,0.25,0.42,-4*1e5,4*1e5,-4*1e5,4*1e5,-4*1e5,4*1e5)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,0.5,0,0.25)
partition = (20,10,10)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,3,5,7,13,15,17,19,25])
add_tag_from_tags!(labels,"neumann1",[22])
add_tag_from_tags!(labels,"neumann2",[24])
add_tag_from_tags!(labels,"neumann3",[26])

Ω = Triangulation(model)
Γ1 = BoundaryTriangulation(model,tags="neumann1")
Γ2 = BoundaryTriangulation(model,tags="neumann2")
Γ3 = BoundaryTriangulation(model,tags="neumann3")

λ(μ) = μ[1]*μ[2]/((1+μ[2])*(1-2*μ[2]))
p(μ) = μ[1]/(2(1+μ[2]))

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

h1(x,μ,t) = VectorValue(0.0,0.0,μ[3]*exp(sin(2*π*t/tf)))
h1(μ,t) = x -> h1(x,μ,t)
h1μt(μ,t) = TransientParamFunction(h1,μ,t)

h2(x,μ,t) = VectorValue(0.0,μ[4]*exp(cos(2*π*t/tf)),0.0)
h2(μ,t) = x -> h2(x,μ,t)
h2μt(μ,t) = TransientParamFunction(h2,μ,t)

h3(x,μ,t) = VectorValue(μ[5]*x[1]*(1+t),0.0,0.0)
h3(μ,t) = x -> h3(x,μ,t)
h3μt(μ,t) = TransientParamFunction(h3,μ,t)

g(x,μ,t) = VectorValue(0.0,0.0,0.0)
g(μ,t) = x -> g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

order = 1
reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)

degree = 2*order
dΩ = Measure(Ω,degree)
dΓ1 = Measure(Γ1,degree)
dΓ2 = Measure(Γ2,degree)
dΓ3 = Measure(Γ3,degree)

res(μ,t,u,v,dΓ1,dΓ2,dΓ3) = (-1)*(∫(v⋅h1μt(μ,t))dΓ1 + ∫(v⋅h2μt(μ,t))dΓ2 + ∫(v⋅h3μt(μ,t))dΓ3)
mass(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
stiffness(μ,t,u,v,dΩ) = ∫(0*v⋅u)dΩ

trian_res = (Γ1,Γ2,Γ3,)
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
ExamplesInterface.run_test(dir,rbsolver,feop,tols)

fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)

end
