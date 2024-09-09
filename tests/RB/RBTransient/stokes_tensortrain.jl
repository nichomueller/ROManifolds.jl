using Gridap
using Gridap.FESpaces
using Gridap.MultiField
using Test
using DrWatson
using Serialization

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

using Mabla.FEM
using Mabla.FEM.Utils
using Mabla.FEM.IndexMaps
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady
using Mabla.RB.RBTransient

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 0.1

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

order = 2
degree = 2*order

a(x,μ,t) = 1+exp(-sin(2π*t/tf)^2*(1-x[2])/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

g0(x,μ,t) = VectorValue(0.0,0.0,0.0)
g0(μ,t) = x->g0(x,μ,t)
g0μt(μ,t) = TransientParamFunction(g0,μ,t)

h(x,μ,t) = VectorValue(abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100),0.0,0.0)
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

energy((du,dp),(v,q),dΩ) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ
energy(dΩ) = (u,v) -> energy(u,v,dΩ)

stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,(u,p),(v,q),dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ) - ∫(v⋅hμt(μ,t))dΓn

for n in (8,10,12,15)
  println("--------------------------------------------------------------------")
  println("TT algorithm, n = $(n)")
  domain = (0,1,0,1,0,1)
  partition = (n,n,n)
  model = TProductModel(domain,partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet0",setdiff(1:26,24))
  add_tag_from_tags!(labels,"neumann",[24])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model.model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  trian_res = (Ω.trian,Γn)
  trian_stiffness = (Ω.trian,)
  trian_mass = (Ω.trian,)

  reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
  trial_u = TransientTrialParamFESpace(test_u,g0μt)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(Ω,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
    trial,test,trian_res,trian_stiffness,trian_mass)

  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0*p*q)dΩ,test_p.space,test_p.space)]
  bblocks = map(CartesianIndices((2,2))) do I
    (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
  end
  coeffs = [1.0 1.0;
            0.0 1.0]
  solver_u = LUSolver()
  solver_p = LUSolver()
  P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
  solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-7,verbose=true)
  odesolver = ThetaMethod(solver,dt,θ)
  lu_odesolver = ThetaMethod(LUSolver(),dt,θ)

  tol = fill(1e-4,4)
  reduction = TTSVDReduction(tol,energy(dΩ);nparams=50)
  rbsolver = RBSolver(odesolver,reduction;nparams_test=10,nparams_res=30,nparams_jac=20)
  test_dir = datadir(joinpath("stokes","3dcube_tensor_train_$(n)"))

  fesnaps,festats = fe_solutions(rbsolver,feop,xh0μ)
  save(test_dir,fesnaps)

  println(festats)

  println("--------------------------------------------------------------------")
  println("Regular algorithm, n = $(n)")

  model = model.model
  Ω = Ω.trian
  dΩ = dΩ.measure

  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
  trial_u = TransientTrialParamFESpace(test_u,g0μt)
  test_p = TestFESpace(model,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
    trial,test,trian_res,trian_stiffness,trian_mass)

  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  tol = 1e-4
  reduction = TransientPODReduction(tol,energy(dΩ);nparams=50)
  rbsolver = RBSolver(odesolver,reduction;nparams_test=10,nparams_res=30,nparams_jac=20)
  test_dir = datadir(joinpath("stokes","3dcube_$(n)"))

  fesnaps = fe_solutions(rbsolver,feop,xh0μ)
  save(test_dir,fesnaps)
end
