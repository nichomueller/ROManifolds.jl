using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Visualization
using Gridap.ODEs
using Test
using DrWatson
using Serialization

using ROM.FEM
using ROM.Utils
using ROM.DofMaps
using ROM.TProduct
using ROM.ParamDataStructures
using ROM.ParamFESpaces
using ROM.ParamSteady
using ROM.ParamODEs

using ROM.RB
using ROM.RB.RBSteady
using ROM.RB.RBTransient

θ = 1
dt = 0.0025
t0 = 0.0
tf = 60*dt

pdomain = [[1,9]*1e10,[0.25,0.42],[-4,4]*1e5,[-4,4]*1e5,[-4,4]*1e5]
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

n = 12
domain = (0,1,0,1/3,0,1/3)
partition = (n,floor(Int,n/3),floor(Int,n/3))
model = TProductDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,3,5,7,13,15,17,19,25])
add_tag_from_tags!(labels,"neumann1",[22])
add_tag_from_tags!(labels,"neumann2",[24])
add_tag_from_tags!(labels,"neumann3",[26])

order = 2
degree = 2*order

Ω = Triangulation(model)
Γ1 = BoundaryTriangulation(model.model,tags="neumann1")
Γ2 = BoundaryTriangulation(model.model,tags="neumann2")
Γ3 = BoundaryTriangulation(model.model,tags="neumann3")

dΩ = Measure(Ω,degree)
dΓ1 = Measure(Γ1,degree)
dΓ2 = Measure(Γ2,degree)
dΓ3 = Measure(Γ3,degree)

λ(μ) = μ[1]*μ[2]/((1+μ[2])*(1-2*μ[2]))
p(μ) = μ[1]/(2(1+μ[2]))

σ(μ,t) = ε -> exp(sin(2*π*t/tf))*(λ(μ)*tr(ε)*one(ε) + 2*p(μ)*ε)
σμt(μ,t) = TransientParamFunction(σ,μ,t)

h1(μ,t) = x -> VectorValue(0.0,0.0,μ[3]*exp(sin(2*π*t/tf)))
h1μt(μ,t) = TransientParamFunction(h1,μ,t)

h2(μ,t) = x -> VectorValue(0.0,μ[4]*exp(cos(2*π*t/tf)),0.0)
h2μt(μ,t) = TransientParamFunction(h2,μ,t)

h3(μ,t) = x -> VectorValue(μ[5]*(1+t),0.0,0.0)
h3μt(μ,t) = TransientParamFunction(h3,μ,t)

g(μ,t) = x -> VectorValue(0.0,0.0,0.0)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(μ) = x -> VectorValue(0.0,0.0,0.0)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫( ε(v) ⊙ (σμt(μ,t)∘ε(u)) )*dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,u,v,dΩ,dΓ1,dΓ2,dΓ3) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,u,v,dΩ) - (∫(v⋅h1μt(μ,t))dΓ1
  + ∫(v⋅h2μt(μ,t))dΓ2 + ∫(v⋅h3μt(μ,t))dΓ3)

energy(du,v) = ∫(v⋅du)dΩ + ∫(∇(v)⊙∇(du))dΩ

trian_res = (Ω.trian,Γ1,Γ2,Γ3)
trian_stiffness = (Ω.trian,)
trian_mass = (Ω.trian,)

reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  trial,test,trian_res,trian_stiffness,trian_mass)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)

tol = fill(1e-4,5)
state_reduction = TTSVDReduction(tol,energy(dΩ);nparams=50)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=30,nparams_jac=20,nparams_djac=1)
test_dir = datadir("elasticity")
create_dir(test_dir)

fesnaps,festats = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
ronline = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,ronline,uh0μ)

x,festats = solution_snapshots(rbsolver,feop,ronline,uh0μ)
perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats,ronline)

module TransientElasticity

using Gridap
using Test
using DrWatson

using ROM

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),nparams_djac=1,sketch=:sprn,unsafe=false
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
  pdomain = (1e10,9*1e10,0.25,0.42,-4*1e5,4*1e5)

  domain = (0,2.5,0,0.4)
  partition = (25,4)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[5]) # bottom face
  dΓn = Measure(Γn,degree)

  λ(μ) = μ[1]*μ[2]/((1+μ[2])*(1-2*μ[2]))
  p(μ) = μ[1]/(2(1+μ[2]))

  σ(μ,t) = ε -> exp(sin(2*π*t/tf))*(λ(μ)*tr(ε)*one(ε) + 2*p(μ)*ε)
  σμt(μ,t) = TransientParamFunction(σ,μ,t)

  h(μ,t) = x -> VectorValue(0.0,μ[3]*exp(sin(2*π*t/tf)))
  hμt(μ,t) = TransientParamFunction(h,μ,t)

  g(μ,t) = x -> VectorValue(0.0,0.0)
  gμt(μ,t) = TransientParamFunction(g,μ,t)

  u0(μ) = x -> VectorValue(0.0,0.0)
  u0μ(μ) = ParamFunction(u0,μ)

  stiffness(μ,t,u,v,dΩ) = ∫( ε(v) ⊙ (σμt(μ,t)∘ε(u)) )*dΩ
  mass(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
  res(μ,t,u,v,dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,u,v,dΩ) - ∫(v⋅hμt(μ,t))dΓn

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  energy(du,v) = ∫(v⋅du)dΩ + ∫(∇(v)⊙∇(du))dΩ

  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7]) # left face, extrema included
  trial = TransientTrialParamFESpace(test,gμt)

  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    state_reduction = TransientReduction(tolrank,energy;nparams,sketch)
  else method == :ttsvd
    tolranks = fill(tolrank,4)
    state_reduction = TTSVDReduction(tolranks,energy;nparams,unsafe)
  end

  θ = 0.5
  dt = 0.01
  t0 = 0.0
  tf = 10*dt
  tdomain = t0:dt:tf

  fesolver = ThetaMethod(LUSolver(),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  ptspace_uniform = TransientParamSpace(pdomain,tdomain;sampling=:uniform)
  feop_uniform = TransientParamLinearFEOperator((stiffness,mass),res,ptspace_uniform,trial,test,domains)
  μon = realization(feop_uniform;nparams=10)
  x,festats = solution_snapshots(rbsolver,feop_uniform,μon,uh0μ)

  for sampling in (:uniform,:halton,:latin_hypercube,:tensorial_uniform)
    println("Running $method test with sampling strategy $sampling")
    ptspace = TransientParamSpace(pdomain,tdomain;sampling)
    feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,trial,test,domains)

    fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
    rbop = reduced_operator(rbsolver,feop,fesnaps)
    x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

    println(perf)
  end

end

main(:pod)
main(:ttsvd)

end
