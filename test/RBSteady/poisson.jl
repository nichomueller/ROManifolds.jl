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

method=:pod
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
unsafe=false

pdomain = (1,10,1,10,1,10)

domain = (0,1,0,1)
partition = (50,50)
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

state_reduction = PODReduction(1e-4)

fesolver = LinearFESolver()
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

pspace = ParamSpace(pdomain)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,domains)
μ = realization(feop;nparams=50)

_fesolver = ParamFESolver(LUSolver())
_rbsolver = RBSolver(_fesolver,state_reduction;nparams_res,nparams_jac)

x,stats = solution_snapshots(rbsolver,feop,μ)
_x,_stats = solution_snapshots(_rbsolver,feop,μ)

u = zero(trial(μ))
x = get_free_dof_values(u)
op = get_algebraic_operator(set_domains(feop))
nlop = parameterize(op,μ)
syscache = ParamSteady.allocate_systemcache(nlop,x)
A = syscache.A
jacobian!(A,nlop,x)

_nlop = lazy_parameterize(op,μ)
_syscache = ParamSteady.allocate_systemcache(_nlop,x)
_A = _syscache.A
jacobian!(_A,_nlop,x)

# jacobian!(A,nlop,x)
# jacobian!(A,nlop.op,nlop.μ,x,nlop.paramcache)
# jacobian_add!(A,nlop.op,nlop.μ,x,nlop.paramcache)
uh = EvaluationFunction(nlop.paramcache.trial,x)
du = get_trial_fe_basis(trial)
v = get_fe_basis(test)
assem = get_param_assembler(nlop.op.op,μ)

dc = nlop.op.op.jac(μ,uh,du,v)
matdata = collect_cell_matrix(trial,test,dc)
assemble_matrix_add!(A,assem,matdata)

using ROManifolds.ParamFESpaces
using ROManifolds.ParamAlgebra
using ROManifolds.ParamSteady
_matdata = ParamAlgebra.get_matdata(_nlop.paramcache)
_matcache = ParamAlgebra.get_matcache(_nlop.paramcache)
_assem = get_assembler(_nlop.op.op)
ParamSteady.assemble_lazy_matrix_add!(_A,_assem,_matdata,_matcache,1)

vc = _matcache[1][1]
cellmat_i = lazy_param_getindex(_matdata[1][1],1)
mat1 = getindex!(vc,cellmat_i,1)
