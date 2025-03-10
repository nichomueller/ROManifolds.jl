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

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  pspace_uniform = ParamSpace(pdomain;sampling=:uniform)
  feop_uniform = LinearParamOperator(res,stiffness,pspace_uniform,trial,test,domains)
  μon = realization(feop_uniform;nparams=10)
  x,festats = solution_snapshots(rbsolver,feop_uniform,μon)

  for sampling in (:uniform,:halton,:latin_hypercube,:tensorial_uniform)
    println("Running $method test with sampling strategy $sampling")
    pspace = ParamSpace(pdomain;sampling)
    feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

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
using ROManifolds.ParamODEs
using ROManifolds.Utils
using Test
using FillArrays
import Gridap.MultiField: BlockMultiFieldStyle
using GridapSolvers
using GridapSolvers.NonlinearSolvers

method=:ttsvd
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
unsafe=false

pdomain = (1,10,1,10,1,10)

domain = (0,1,0,1)
partition = (3,3)
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

a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = TransientTrialParamFESpace(test,gμt)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

state_reduction = Reduction(fill(1e-4,3),energy;nparams=5)
# state_reduction = TransientReduction(1e-4,energy;nparams=50)

# coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
# state_reduction = SupremizerReduction(coupling,1e-4,energy;nparams=50,sketch=:sprn)
# ttcoupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
# state_reduction = SupremizerReduction(ttcoupling,fill(1e-4,4),energy;nparams=50)
θ = 1
dt = 0.01
t0 = 0.0
tf = 10*dt
tdomain = t0:dt:tf

fesolver = ThetaMethod(LUSolver(),dt,θ)

rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

ptspace = TransientParamSpace(pdomain,tdomain)
feop = TransientParamLinearOperator((stiffness,mass),res,ptspace,trial,test,domains)

fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=2)
x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)
x,festats = solution_snapshots(rbsolver,feop,μon,uh0μ)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

using ROManifolds.DofMaps
using ROManifolds.ParamAlgebra
using ROManifolds.RBSteady
using ROManifolds.RBTransient

op,r = rbop,μon
x̂ = zero_free_values(get_trial(op)(r))
u = x̂

nlop = parameterize(op,r)
syscache = allocate_systemcache(nlop,x̂)
paramcache = nlop.paramcache
shift!(r,dt*(θ-1))
update_paramcache!(paramcache,nlop.op,r)
shift!(r,dt*(1-θ))

# A = jacobian(nlop,x̂)
# b = residual(nlop,x̂)

np = num_params(r)
hr_time_ids = RBTransient.get_common_time_domain(op.rhs)
hr_param_time_ids = range_1d(1:np,hr_time_ids,np)
hr_uh = RBTransient._make_hr_uh_from_us(op.op,u,paramcache.trial,hr_param_time_ids)

v = get_fe_basis(test)
trian_res = get_domains_res(op.op)
μ = get_params(r)
hr_t = view(get_times(r),hr_time_ids)
dc = get_res(op.op)(μ,hr_t,hr_uh,v)

strian = trian_res[1]
rhs_strian = op.rhs[strian]
vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian,hr_param_time_ids)
cellvec,cellidsrows,icells,locations = vecdata
style = RBTransient.TransientHRStyle(rhs_strian)

# U = param_getindex(trial(r),1)
# w0 = zero(U)
# wh0 = TransientCellField(w0,(w0,))
# μ1,t1 = μon.params[1].params,10*dt
# _dΩ = dΩ.measure
# _res1(v) = ∫(a(μ1,t1)*∇(v)⋅∇(wh0))_dΩ + ∫(v*∂t(wh0))_dΩ - ∫(f(μ1,t1)*v)_dΩ - ∫(h(μ1,t1)*v)dΓn
# B1 = assemble_vector(_res1,test.space)
# t2 = dt
# _res2(v) = ∫(a(μ1,t2)*∇(v)⋅∇(wh0))_dΩ + ∫(v*∂t(wh0))_dΩ - ∫(f(μ1,t2)*v)_dΩ - ∫(h(μ1,t2)*v)dΓn
# B2 = assemble_vector(_res2,test.space)

# Bselect = [B1[4],B1[5]]

b = syscache.b
b_strian = b.fecache[strian]
# assemble_hr_vector_add!(b_strian,style,cellvec,cellidsrows,icells,locations)

cellvec,cellidsrows,icells,locations = vecdata
add! = RBTransient.AddTransientHREntriesMap(style,locations)
add_cache = return_cache(add!,b_strian,vals,rows)

cell = 1
vals,rows = cellvec[cell],cellidsrows[cell]

li = 4
i = rows[li]
RBTransient.get_hr_param_entry!(add_cache,vals,locations,li)
@assert add_cache ≈ map(x -> x[li],vals.data)
RBTransient.add_hr_entry!(+,b_strian,add_cache,locations)

CIAO
# op,r = rbop,μon
# trial = get_trial(op)(r)
# x̂ = zero_free_values(trial)
# u = x̂

# nlop = parameterize(op,r)
# syscache = allocate_systemcache(nlop,x̂)
# paramcache = nlop.paramcache
# shift!(r,dt*(θ-1))
# update_paramcache!(paramcache,nlop.op,r)
# shift!(r,dt*(1-θ))
# np = num_params(r)
# hr_time_ids = RBTransient.get_common_time_domain(op.lhs)
# hr_param_time_ids = range_1d(1:np,hr_time_ids,np)
# hr_uh = RBTransient._make_hr_uh_from_us(op.op,u,paramcache.trial,hr_param_time_ids)

# v = get_fe_basis(test)
# du = get_trial_fe_basis(test)
# trian_lhs = get_domains_jac(op.op)[2]
# μ = get_params(r)
# hr_t = view(get_times(r),hr_time_ids)
# dc = get_jacs(op.op)[2](μ,hr_t,hr_uh,du,v)

# strian = trian_lhs[1]
# lhs_strian = op.lhs[2][strian]
# matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian,hr_param_time_ids)
# cellmat,cellidsrows,cellidscols,icells,locations = matdata
# style = RBTransient.TransientHRStyle(lhs_strian)

# A = syscache.A
# A_strian = A.fecache[2][strian]

# assemble_hr_matrix_add!(A_strian,style,cellmat,cellidsrows,cellidscols,icells,locations)
