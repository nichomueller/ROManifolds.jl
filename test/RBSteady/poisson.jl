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

pdomain = (1,10,1,10,1,10)

domain = (0,1,0,1)
partition = (10,10)
model = CartesianDiscreteModel(domain,partition)

order = 2
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)

a(μ) = x -> μ[1]*exp(-x[1])
aμ(μ) = ParamFunction(a,μ)

g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
gμ(μ) = ParamFunction(g,μ)

stiffness(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
res(μ,(u,p),(v,q),dΩ) = stiffness(μ,(u,p),(v,q),dΩ)

trian_res = (Ω,)
trian_stiffness = (Ω,)
domains = FEDomains(trian_res,trian_stiffness)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

state_reduction = Reduction(1e-4)

fesolver = LinearFESolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction)

pspace = ParamSpace(pdomain)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,domains)

μ = realization(feop;nparams=10)
x,festats = solution_snapshots(rbsolver,feop,μ)

U = trial(μ)
v,q = get_fe_basis(U)
du,dp = get_trial_fe_basis(U)
u,p = zero(U)

m1 = ∫(aμ(μ)*∇(v)⊙∇(du))dΩ
m2 = ∫(dp*(∇⋅(v)))dΩ
m3 = ∫(q*(∇⋅(du)))dΩ

(m1 + m2 + m3)[Ω][1]

v1 = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ
v2 = ∫(p*(∇⋅(v)))dΩ
v3 = ∫(q*(∇⋅(u)))dΩ

(v1 + v2 + v3)[Ω][1]

k = Broadcasting(+)
α = dc1[Ω][1]
β = dc3[Ω][1]

k = Fields.BroadcastingFieldOpMap(+)
cache = return_cache(k,α,β)

N = 1
A = eltype(α)
B = eltype(β)
fi = testvalue(A)
gi = testvalue(B)
ci = return_cache(k,fi,gi)
hi = evaluate!(ci,k,fi,gi)
m = Fields.ZeroBlockMap()
aa = Array{typeof(hi),N}(undef,size(α.array))
bb = Array{typeof(ci),N}(undef,size(α.array))
zf = Array{typeof(return_cache(m,fi,gi))}(undef,size(α.array))
zg = Array{typeof(return_cache(m,gi,fi))}(undef,size(α.array))
t = map(|,α.touched,β.touched)

i = 1
_fi = α.array[i]
zg[i] = return_cache(m,gi,_fi)
_gi = evaluate!(zg[i],m,gi,_fi)
bb[i] = return_cache(k,_fi,_gi)

i = 3
_gi = β.array[i]
zf[i] = return_cache(m,fi,_gi)
_fi = evaluate!(zf[i],m,fi,_gi)
bb[i] = return_cache(k,_fi,_gi)

# return_cache(m,gi,_fi) ok
# return_cache(m,fi,_gi) not ok

hh,ff = ParamDataStructures._parameterize(fi,_gi)
hi = testitem(hh)
fi = testitem(ff)
ci = return_cache(m,hi,fi)
ri = evaluate!(ci,m,hi,fi)
c = Vector{typeof(ci)}(undef,param_length(f))
data = Vector{typeof(ri)}(undef,param_length(f))
