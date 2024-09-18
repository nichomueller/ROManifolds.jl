using Gridap
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Test
using DrWatson
using Serialization

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

h(x,μ,t) = VectorValue(x[2]*(1-x[2])*abs(1-cos(9*π*t/(5*tf))+μ[3]*sin(μ[2]*9*π*t/(5*tf))/100),0.0,0.0)
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

const Re = 100
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

jac_lin_u(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ
mass_u(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
res_lin_u(μ,t,u,v,dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + jac_lin_u(μ,t,u,v,dΩ) - ∫(v⋅hμt(μ,t))dΓn

jac_nlin_u(μ,t,u,du,v,dΩ) = dc(u,du,v,dΩ)
res_nlin_u(μ,t,u,v,dΩ) = c(u,v,dΩ)

energy_u(u,v,dΩ) = ∫(v⋅u)dΩ + ∫(∇(v)⊙∇(u))dΩ
energy_u(dΩ) = (u,v) -> energy_u(u,v,dΩ)

for n in (8,10,12,15)
  println("--------------------------------------------------------------------")
  println("TT algorithm, n = $(n)")
  domain = (0,1,0,2/3,0,2/3)
  partition = (n,floor(2*n/3),floor(2*n/3))
  model = TProductModel(domain,partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet0",setdiff(1:26,22))
  add_tag_from_tags!(labels,"neumann",[22])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model.model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  trian_res = (Ω.trian,Γn)
  trian_jac = (Ω.trian,)
  trian_djac = (Ω.trian,)

  reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
  trial_u = TransientTrialParamFESpace(test_u,g0μt)

  feop_lin_u = TransientParamLinearFEOperator((jac_lin_u,mass_u),res_lin_u,ptspace,
    trial_u,test_u,trian_res,trian_jac,trian_djac)
  feop_nlin_u = TransientParamFEOperator(res_nlin_u,jac_nlin_u,ptspace,
    trial_u,test_u,trian_jac,trian_jac)
  feop_u = LinNonlinTransientParamFEOperator(feop_lin_u,feop_nlin_u)

  fesolver = ThetaMethod(NewtonRaphsonSolver(LUSolver(),1e-10,20),dt,θ)

  tol = fill(1e-5,5)
  reduction = TTSVDReduction(tol,energy_u(dΩ);nparams=50)
  rbsolver = RBSolver(fesolver,reduction;nparams_test=10,nparams_res=20,nparams_jac=10,nparams_djac=1)
  test_dir = datadir(joinpath("navier-stokes","3dcube_tensor_train_$(n)"))

  fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
  rbop = reduced_operator(rbsolver,feop_u,fesnaps[1])

  println("--------------------------------------------------------------------")
  println("Regular algorithm, n = $(n)")

  model = model.model
  Ω = Ω.trian
  dΩ = dΩ.measure

  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
  trial_u = TransientTrialParamFESpace(test_u,g0μt)

  feop_lin_u = TransientParamLinearFEOperator((jac_lin_u,mass_u),res_lin_u,ptspace,
    trial_u,test_u,trian_res,trian_jac,trian_djac)
  feop_nlin_u = TransientParamFEOperator(res_nlin_u,jac_nlin_u,ptspace,
    trial_u,test_u,trian_jac,trian_jac)
  feop_u = LinNonlinTransientParamFEOperator(feop_lin_u,feop_nlin_u)

  tol = 1e-4
  reduction = TransientPODReduction(tol,energy_u(dΩ);nparams=50)
  rbsolver = RBSolver(fesolver,reduction;nparams_test=10,nparams_res=30,nparams_jac=20,nparams_djac=1)
  test_dir = datadir(joinpath("navier-stokes","3dcube_$(n)"))

  # fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
  fesnaps = change_index_map(fesnaps,TrivialIndexMap)
  rbop = reduced_operator(rbsolver,feop_u,fesnaps[1])
end

n = 8
domain = (0,1,0,2/3,0,2/3)
partition = (n,floor(2*n/3),floor(2*n/3))
model = TProductModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet0",setdiff(1:26,22))
add_tag_from_tags!(labels,"neumann",[22])

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model.model,tags=["neumann"])
dΓn = Measure(Γn,degree)

trian_res = (Ω.trian,Γn)
trian_jac = (Ω.trian,)
trian_djac = (Ω.trian,)

reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,g0μt)

feop_lin_u = TransientParamLinearFEOperator((jac_lin_u,mass_u),res_lin_u,ptspace,
  trial_u,test_u,trian_res,trian_jac,trian_djac)
feop_nlin_u = TransientParamFEOperator(res_nlin_u,jac_nlin_u,ptspace,
  trial_u,test_u,trian_jac,trian_jac)
feop_u = LinNonlinTransientParamFEOperator(feop_lin_u,feop_nlin_u)

fesolver = ThetaMethod(NewtonRaphsonSolver(LUSolver(),1e-10,20),dt,θ)

tol = fill(1e-5,5)
reduction = TTSVDReduction(tol,energy_u(dΩ);nparams=50)
rbsolver = RBSolver(fesolver,reduction;nparams_test=10,nparams_res=20,nparams_jac=10,nparams_djac=1)
test_dir = datadir(joinpath("navier-stokes","3dcube_tensor_train_$(n)"))

fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
# rbop = reduced_operator(rbsolver,feop_u,fesnaps[1])
red_trial,red_test = reduced_fe_space(rbsolver,feop_u,fesnaps[1])
op = get_algebraic_operator(feop_u)
pop = TransientPGOperator(op,red_trial,red_test)
# reduced_operator(rbsolver,op,red_trial,red_test,fesnaps[1])

# # reduced_operator(rbsolver,get_linear_operator(pop),fesnaps[1])
# poplin = get_linear_operator(pop)
# # reduced_jacobian_residual(rbsolver,poplin,fesnaps[1])
# jacs,ress = jacobian_and_residual(rbsolver,poplin,fesnaps[1])
# red_jac = reduced_jacobian(RBSteady.get_jacobian_reduction(rbsolver),poplin,jacs)
# red_res = reduced_residual(RBSteady.get_residual_reduction(rbsolver),poplin,ress)
# reduced_operator(rbsolver,get_nonlinear_operator(pop),fesnaps[1])
popnlin = get_nonlinear_operator(pop)
# reduced_jacobian_residual(rbsolver,poplin,fesnaps[1])
nljacs,nlress = jacobian_and_residual(rbsolver,popnlin,fesnaps[1])
# red_jac = reduced_jacobian(RBSteady.get_jacobian_reduction(rbsolver),popnlin,nljacs)
# red_res = reduced_residual(RBSteady.get_residual_reduction(rbsolver),popnlin,ress)
