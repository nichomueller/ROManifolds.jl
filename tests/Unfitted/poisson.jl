module PoissonEmbedded

using Gridap
using GridapEmbedded
using ROM

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
  Ωact = Triangulation(cutgeo,ACTIVE)
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
  b(μ,u,v,dΩ,dΓ) = a(μ,u,v,dΩ,dΓ) - ∫( v*fμ(μ) )dΩ - ∫( νμ(μ)*(nΓ⋅∇(v))*gμ(μ) )dΓ

  trian_a = (Ω,Γ)
  trian_b = (Ω,Γ)
  domains = FEDomains(trian_b,trian_a)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    test = TestFESpace(Ωact,reffe;conformity=:H1)
    state_reduction = PODReduction(tolrank,energy;nparams,sketch)
  else method == :ttsvd
    test = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1)
    tolranks = fill(tolrank,3)
    state_reduction = Reduction(tolranks,energy;nparams,unsafe)
  end

  trial = ParamTrialFESpace(test)
  feop = LinearParamFEOperator(res,stiffness,pspace,trial,test,domains)

  fesolver = LinearFESolver(LUSolver())
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  # offline
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  # online
  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  # test
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)
  println(perf)
end

main(:pod)
main(:ttsvd)

end


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
Ωact = Triangulation(cutgeo,ACTIVE)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
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
b(μ,u,v,dΩ,dΓ) = a(μ,u,v,dΩ,dΓ) - ∫( v*fμ(μ) )dΩ - ∫( νμ(μ)*(nΓ⋅∇(v))*gμ(μ) )dΓ
domains = FEDomains((Ω,Γ),(Ω,Γ))

reffe = ReferenceFE(lagrangian,Float64,order)
test = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1)
trial = ParamTrialFESpace(test,gμ)
feop = LinearParamFEOperator(b,a,pspace,trial,test,domains)

tol = 1e-4
energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
state_reduction = TTSVDReduction(tol,energy;nparams=100)
fesolver = LinearFESolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction)

# offline
fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

# online
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)

# test
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats,μon)

using ROM.RBSteady
rbsnaps = RBSteady.to_snapshots(rbop.trial,x̂,μon)

r1 = get_realization(fesnaps)[1]
S1 = get_param_data(fesnaps)[1]
plt_dir = datadir("plts")
create_dir(plt_dir)
uh1 = OrderedFEFunction(param_getindex(trial(r1),1),S1)
writevtk(Ω,joinpath(plt_dir,"sol.vtu"),cellfields=["uh"=>uh1])

# gridap
νeval(x) = sum(μ.params[1])
feval(x) = μ.params[1][1]*x[1] - μ.params[1][2]*x[2]
geval(x) = μ.params[1][3]*sum(x)
lhs(u,v) = ∫( νeval*∇(v)⋅∇(u) )dΩ - ∫( νeval*v*(nΓ⋅∇(u)) - νeval*(nΓ⋅∇(v))*u )dΓ
rhs(v) = ∫( v*feval )dΩ + ∫( νeval*(nΓ⋅∇(v))*geval )dΓ
V = TestFESpace(Ωact,reffe;conformity=:H1)
U = TrialFESpace(V,geval)
feop = AffineFEOperator(lhs,rhs,U,V)
uh = solve(feop)

using DrWatson
using ROM.ParamDataStructures
r1 = get_realization(fesnaps)[1]
S1 = get_param_data(fesnaps)[1]
plt_dir = datadir("plts")
create_dir(plt_dir)
uh1 = FEFunction(param_getindex(trial(r1),1),S1)
writevtk(Ω,joinpath(plt_dir,"sol.vtu"),cellfields=["uhapp"=>uh1,"uh"=>uh,"eh"=>uh1-uh])
writevtk(Ωact,joinpath(plt_dir,"solact.vtu"),cellfields=["uhapp"=>uh1,"uh"=>uh,"eh"=>uh1-uh])


# agfem tests

R = 0.5
L = 0.8*(2*R)
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(L,0.0)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R

n = 30
partition = (n,n)
bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
dp = pmax - pmin

cutgeo = cut(bgmodel,geo3)

strategy = AggregateAllCutCells()
aggregates = compute_bgcell_to_inoutcut(bgmodel,geo)

Ω_bg = Triangulation(bgmodel)
Ω_act = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)

n_Γ = get_normal_vector(Γ)

order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

model = get_active_model(Ω_act)
Vstd = FESpace(Ω_act,FiniteElements(PhysicalDomain(),model,lagrangian,Float64,order))

V = AgFEMSpace(Vstd,aggregates)

f = Vstd
# Triangulation made of active cells
trian_a = get_triangulation(f)
bgcell_to_bgcellin = aggregates
bgcell_to_gcell=1:length(bgcell_to_bgcellin)

# Build root cell map (i.e. aggregates) in terms of active cell ids
D = num_cell_dims(trian_a)
glue = get_glue(trian_a,Val(D))
acell_to_bgcell = glue.tface_to_mface
bgcell_to_acell = glue.mface_to_tface
acell_to_bgcellin = collect(lazy_map(Reindex(bgcell_to_bgcellin),acell_to_bgcell))
acell_to_acellin = collect(lazy_map(Reindex(bgcell_to_acell),acell_to_bgcellin))
acell_to_gcell = lazy_map(Reindex(bgcell_to_gcell),acell_to_bgcell)

# Build shape funs of g by replacing local funs in cut cells by the ones at the root
# This needs to be done with shape functions in the physical domain
# otherwise shape funs in cut and root cells are the same
using Gridap.CellData
using Gridap.FESpaces
shfns_g = get_fe_basis(f)
dofs_g = get_fe_dof_basis(f)
acell_phys_shapefuns_g = get_array(change_domain(shfns_g,PhysicalDomain()))
acell_phys_root_shapefuns_g = lazy_map(Reindex(acell_phys_shapefuns_g),acell_to_acellin)
root_shfns_g = GenericCellField(acell_phys_root_shapefuns_g,trian_a,PhysicalDomain())

# Compute data needed to compute the constraints
dofs_f = get_fe_dof_basis(f)
shfns_f = get_fe_basis(f)
acell_to_coeffs = dofs_f(root_shfns_g)
acell_to_proj = dofs_g(shfns_f)
acell_to_dof_ids = get_cell_dof_ids(f)

aggdof_to_fdof, aggdof_to_dofs, aggdof_to_coeffs = GridapEmbedded.AgFEM._setup_agfem_constraints(
  num_free_dofs(f),
  acell_to_acellin,
  acell_to_dof_ids,
  acell_to_coeffs,
  acell_to_proj,
  acell_to_gcell)

mDOF_to_DOF = V.mDOF_to_DOF
v = fill(false,31,31)
v[mDOF_to_DOF] .= true
