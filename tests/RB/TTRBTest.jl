using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.FESpaces
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools
using Test
using DrWatson
using Mabla.FEM
using Mabla.RB
using LinearAlgebra
using SparseArrays

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

for n in (5,10,20)
  domain = (0,1,0,1)
  partition = (n,n)
  model = CartesianDiscreteModel(domain, partition)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
  add_tag_from_tags!(labels,"neumann",[7])

  order = 1
  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = TransientParamFunction(a,μ,t)

  f(x,μ,t) = 1.
  f(μ,t) = x->f(x,μ,t)
  fμt(μ,t) = TransientParamFunction(f,μ,t)

  h(x,μ,t) = abs(cos(t/μ[3]))
  h(μ,t) = x->h(x,μ,t)
  hμt(μ,t) = TransientParamFunction(h,μ,t)

  g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  g(μ,t) = x->g(x,μ,t)
  gμt(μ,t) = TransientParamFunction(g,μ,t)

  u0(x,μ) = 0
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = ParamFunction(u0,μ)

  res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
  jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
  jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

  induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

  trian_res = (Ω,Γn)
  trian_jac = (Ω,)
  trian_jac_t = (Ω,)

  reffe = ReferenceFE(lagrangian,Float64,order)
  tt_test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"],vector_type=TTVector{1,Float64})
  tt_trial = TransientTrialParamFESpace(tt_test,gμt)
  _tt_feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,tt_trial,tt_test)
  tt_feop = FEOperatorWithTrian(_tt_feop,trian_res,trian_jac,trian_jac_t)
  tt_uh0μ(μ) = interpolate_everywhere(u0μ(μ),tt_trial(μ,t0))
  fesolver = ThetaMethod(LUSolver(),dt,θ)

  ϵ = 1e-4
  tt_rbsolver = TTRBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=5,nsnaps_mdeim=20)
  tt_test_dir = get_test_directory(tt_rbsolver,dir=datadir(joinpath("heateq","tt_toy_$(n)_h1")))

  tt_fesnaps,festats = ode_solutions(tt_rbsolver,tt_feop,tt_uh0μ)
  tt_rbop = reduced_operator(tt_rbsolver,tt_feop,tt_fesnaps)
  tt_rbsnaps,tt_rbstats = solve(tt_rbsolver,tt_rbop,tt_fesnaps)
  tt_results = rb_results(tt_rbsolver,tt_feop,tt_fesnaps,tt_rbsnaps,festats,tt_rbstats)
  save(tt_test_dir,tt_fesnaps)
  save(tt_test_dir,tt_rbop)
  save(tt_test_dir,tt_results)
  # tt_results = load_solve(tt_rbsolver,dir=tt_test_dir)
  println(RB.space_time_error(tt_results))
  println(RB.speedup(tt_results))

  #
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = TransientTrialParamFESpace(test,gμt)
  _feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,trial,test)
  feop = FEOperatorWithTrian(_feop,trian_res,trian_jac,trian_jac_t)
  rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=5,nsnaps_mdeim=20)
  test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","toy_$(n)_h1")))

  fesnaps = RB.to_standard_snapshots(tt_fesnaps)
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
  results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)
  println(RB.space_time_error(results))
  println(RB.speedup(results))
  save(test_dir,fesnaps)
  save(test_dir,rbop)
  save(test_dir,results)
  # results = load_solve(rbsolver,dir=test_dir)
end

domain = (0,1,0,1)
partition = (20,20)
model = CartesianDiscreteModel(domain, partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,8])
add_tag_from_tags!(labels,"neumann",[7])

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=["neumann"])
dΓn = Measure(Γn,degree)

a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn
jac(μ,t,u,du,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

induced_norm(du,v) = ∫(∇(v)⋅∇(du))dΩ

trian_res = (Ω,Γn)
trian_jac = (Ω,)
trian_jac_t = (Ω,)

reffe = ReferenceFE(lagrangian,Float64,order)
tt_test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"],vector_type=TTVector{1,Float64})
tt_trial = TransientTrialParamFESpace(tt_test,gμt)
_tt_feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,tt_trial,tt_test)
tt_feop = FEOperatorWithTrian(_tt_feop,trian_res,trian_jac,trian_jac_t)
tt_uh0μ(μ) = interpolate_everywhere(u0μ(μ),tt_trial(μ,t0))
fesolver = ThetaMethod(LUSolver(),dt,θ)

function _ode_solutions(
  fesolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  nparams = 55
  sol = solve(fesolver,op,uh0;nparams)
  odesol = sol.odesol
  realization = odesol.r

  stats = @timed begin
    values = collect(odesol)
  end
  snaps = Snapshots(values,realization)
  cs = ComputationalStats(stats,nparams)
  return snaps,cs
end

tt_fesnaps,festats = _ode_solutions(fesolver,tt_feop,tt_uh0μ)
fesnaps = RB.to_standard_snapshots(tt_fesnaps)

for ϵ in (1e-2,1e-3,1e-4)
  tt_rbsolver = TTRBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=5,nsnaps_mdeim=20)
  tt_test_dir = get_test_directory(tt_rbsolver,dir=datadir(joinpath("heateq","tt_toy_$(ϵ)_h1")))
  tt_rbop = reduced_operator(tt_rbsolver,tt_feop,tt_fesnaps)
  tt_rbsnaps,tt_rbstats = solve(tt_rbsolver,tt_rbop,tt_fesnaps)
  tt_results = rb_results(tt_rbsolver,tt_feop,tt_fesnaps,tt_rbsnaps,festats,tt_rbstats)
  println(num_free_dofs(tt_rbop.op.test))
  println(RB.space_time_error(tt_results))
  println(RB.speedup(tt_results))
  save(tt_test_dir,tt_fesnaps)
  save(tt_test_dir,tt_rbop)
  save(tt_test_dir,tt_results)

  #
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = TransientTrialParamFESpace(test,gμt)
  _feop = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,ptspace,trial,test)
  feop = FEOperatorWithTrian(_feop,trian_res,trian_jac,trian_jac_t)
  rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=5,nsnaps_mdeim=20)
  test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","toy_$(ϵ)_h1")))

  rbop = reduced_operator(rbsolver,feop,fesnaps)
  rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
  results = rb_results(rbsolver,feop,fesnaps,rbsnaps,festats,rbstats)
  println(num_free_dofs(rbop.op.test))
  println(RB.space_time_error(results))
  println(RB.speedup(results))
  save(test_dir,fesnaps)
  save(test_dir,rbop)
  save(test_dir,results)
end
