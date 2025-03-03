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
partition = (20,20)
model = CartesianDiscreteModel(domain,partition)

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

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = ParamTrialFESpace(test,gμ)

state_reduction = Reduction(1e-4)

fesolver = LinearFESolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction)

pspace = ParamSpace(pdomain)
feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,domains)

μ = realization(feop;nparams=10)
x,festats = solution_snapshots(rbsolver,feop,μ)

U = trial(μ)
fv = zero_free_values(U)
dv = zero_dirichlet_values(U)
cell_vals = scatter_free_and_dirichlet_values(U,fv,dv)
zero(U)

cell_dof_ids = get_cell_dof_ids(U)
# lazy_map(Broadcasting(PosNegReindex(free_values,dirichlet_values)),cell_dof_ids)
k = Broadcasting(PosNegReindex(fv,dv))
c = return_cache(k,cell_dof_ids[1])

gi = cell_vals.maps[1]
fi = map(fj -> fj[1],cell_vals.args)
cache = return_cache(gi,fi...)
vi = evaluate(gi, fi...)

using Gridap.MultiField
U = TrialFESpace(test,x->x[2])
space = MultiFieldFESpace([U,U];style=BlockMultiFieldStyle())

dir_values = get_dirichlet_dof_values(space)
dir_values_scratch = zero_dirichlet_values(space)
cell_vals = FESpaces._cell_vals(space,x->x[2])

v = get_fe_basis(test)
du = get_trial_fe_basis(test)
∫(aμ(μ)*∇(v)⋅∇(du))dΩ
u = zero(trial(μ))
∫(aμ(μ)*∇(v)⋅∇(u))dΩ

∇(v)⋅∇(u)
dv,du = ∇(v),∇(u)
dv⋅du
x = get_cell_points(Ω)
du(x)

xd = get_data(x)
fd = get_data(du)
f1 = lazy_map(evaluate,fd.args[1],xd)
# f2 = lazy_map(evaluate,fd.args[2],xd)
i_to_values = fd.args[2].args[1]
i_to_basis = fd.args[2].args[2]
i_to_basis_x = lazy_map(evaluate,i_to_basis,xd)
lazy_map(LinearCombinationMap(:),i_to_values,i_to_basis_x)
