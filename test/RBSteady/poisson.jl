module PoissonEquation

using Gridap
using ROManifolds

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,unsafe=false
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
  pdomain = (1,10,1,10,1,10)

  domain = (0,1,0,1)
  partition = (20,20)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 1
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  a(μ) = x -> exp(-x[1]/sum(μ))
  aμ(μ) = ParamFunction(a,μ)

  f(μ) = x -> 1.
  fμ(μ) = ParamFunction(f,μ)

  g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
  gμ(μ) = ParamFunction(g,μ)

  h(μ) = x -> abs(cos(μ[3]*x[2]))
  hμ(μ) = ParamFunction(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    state_reduction = PODReduction(tolrank,energy;nparams,sketch)
  else method == :ttsvd
    tolranks = fill(tolrank,3)
    state_reduction = Reduction(tolranks,energy;nparams,unsafe)
  end

  fesolver = LinearFESolver(LUSolver())
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  pspace_uniform = ParamSpace(pdomain;sampling=:uniform)
  feop_uniform = LinearParamFEOperator(res,stiffness,pspace_uniform,trial,test,domains)
  μon = realization(feop_uniform;nparams=10)
  x,festats = solution_snapshots(rbsolver,feop_uniform,μon)

  for sampling in (:uniform,:halton,:latin_hypercube,:tensorial_uniform)
    println("Running $method test with sampling strategy $sampling")
    pspace = ParamSpace(pdomain;sampling)
    feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,domains)

    fesnaps, = solution_snapshots(rbsolver,feop)
    rbop = reduced_operator(rbsolver,feop,fesnaps)
    x̂,rbstats = solve(rbsolver,rbop,μon)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

    println(perf)
  end

end

main(:pod)
main(:ttsvd)

end


using Gridap
using Gridap.Algebra
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.CellData
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using Gridap.ODEs
using ROManifolds
using ROManifolds.ParamDataStructures
using ROManifolds.ParamSteady
using Test
using FillArrays
import Gridap.MultiField: BlockMultiFieldStyle
using GridapSolvers
using GridapSolvers.NonlinearSolvers

method=:pod
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
unsafe=false

pdomain = (1,10,-1,5,1,2)

domain = (0,1,0,1)
partition = (20,20)
if method==:ttsvd
  model = TProductDiscreteModel(domain,partition)
else
  model = CartesianDiscreteModel(domain,partition)
end

order = 2
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

Re = 100
a(μ) = x -> μ[1]/Re*exp(-x[1])
aμ(μ) = ParamFunction(a,μ)

g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
gμ(μ) = ParamFunction(g,μ)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

jac_lin(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
res_lin(μ,(u,p),(v,q),dΩ) = jac_lin(μ,(u,p),(v,q),dΩ)

res_nlin(μ,(u,p),(v,q),dΩ) = c(u,v,dΩ)
jac_nlin(μ,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

trian_res = (Ω,)
trian_jac = (Ω,)
domains_lin = FEDomains(trian_res,trian_jac)
domains_nlin = FEDomains(trian_res,trian_jac)

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

state_reduction = PODReduction(1e-4)

fesolver = NonlinearFESolver(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true))
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

pspace = ParamSpace(pdomain)
feop_lin = LinearParamFEOperator(res_lin,jac_lin,pspace,trial,test,domains_lin)
feop_nlin = ParamFEOperator(res_nlin,jac_nlin,pspace,trial,test,domains_nlin)
feop = LinearNonlinearParamFEOperator(feop_lin,feop_nlin)

fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;sampling=:uniform,nparams=10)
x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

using ROManifolds.ParamAlgebra
using ROManifolds.ParamSteady
μ = realization(feop)
U = trial(μ)
uh = zero(U)
x = get_free_dof_values(uh)
op = get_algebraic_operator(set_domains(feop))
nlop = parameterize(op,μ)
syscache = allocate_systemcache(nlop,x)

# solve!(x,fesolver.nls,nlop,syscache)
A,b = syscache.A,syscache.b
residual!(b,nlop,x)
jacobian!(A,nlop,x)
