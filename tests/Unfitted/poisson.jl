module PoissonEmbedded

using Gridap
using GridapEmbedded
using ROM

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
Ω_out = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩ_out = Measure(Ω_out,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)

nΓ = get_normal_vector(Γ)
nΓg = get_normal_vector(Γg)

const γd = 10.0
const γg = 0.1
const h = dp[1]/n

ν(μ) = x->sum(μ)
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

g(μ) = x->μ[3]*sum(x)
gμ(μ) = ParamFunction(g,μ)

# non-symmetric formulation

a(μ,u,v,dΩ,dΓ) = ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ - ∫( νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
b(μ,u,v,dΩ,dΓ) = a(μ,u,v,dΩ,dΓ) - ∫( v*fμ(μ) )dΩ + ∫( νμ(μ)*(nΓ⋅∇(v))*gμ(μ) )dΓ

domains = FEDomains((Ω,Ω_out,Γ),(Ω,Ω_out,Γ))

reffe = ReferenceFE(lagrangian,Float64,order)

test = TProductFESpace(Ωbg,reffe,conformity=:H1)
trial = ParamTrialFESpace(test)
feop = LinearParamFEOperator(b,a,pspace,trial,test,domains)

fesolver = LinearFESolver(LUSolver())

tol = 1e-4
energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
state_reduction = TTSVDReduction(tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=80,nparams_jac=80)

fesnaps,festats = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)


tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  n=20,method=:pod;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,unsafe=false
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

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

  domain = (0,1,0,1)
  partition = (n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(domain,partition)
  else
    bgmodel = CartesianDiscreteModel(domain,partition)
  end

  order = 1
  degree = 2*order

  cutgeo = cut(bgmodel,geo)
  Ωbg = Triangulation(bgmodel)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  nΓ = get_normal_vector(Γ)

  ν(μ) = x->sum(μ)
  νμ(μ) = ParamFunction(ν,μ)

  f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
  fμ(μ) = ParamFunction(f,μ)

  g(μ) = x->μ[3]*sum(x)
  gμ(μ) = ParamFunction(g,μ)

  # non-symmetric formulation

  a(μ,u,v,dΩ,dΓ) = ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ - ∫( νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
  b(μ,u,v,dΩ,dΓ) = a(μ,u,v,dΩ,dΓ) - ∫( v*fμ(μ) )dΩ + ∫( νμ(μ)*(nΓ⋅∇(v))*gμ(μ) )dΓ

  trian_a = (Ω,Γ)
  trian_b = (Ω,Γ)
  domains = FEDomains(trian_b,trian_a)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,domains)

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    state_reduction = PODReduction(tolrank,energy;nparams,sketch)
  else method == :ttsvd
    tolranks = fill(tolrank,3)
    state_reduction = Reduction(tolranks,energy;nparams,unsafe)
  end

  fesolver = LinearFESolver(LUSolver())
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  # offline
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  # online
  μon = realization(feop_uniform;nparams=10)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  # test
  x,festats = solution_snapshots(rbsolver,feop_uniform,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)
  println(perf)
end

main(:pod)
main(:ttsvd)

end
