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

function _jacobian_and_residual(solver::RBSolver,op,s)
  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,RBSteady.jac_params(solver))
  sres = select_snapshots(s,RBSteady.res_params(solver))
  us_jac,us_res = (get_values(sjac),),(get_values(sres),)
  r_jac,r_res = get_realization(sjac),get_realization(sres)
  A = jacobian(fesolver,op,r_jac,us_jac)
  b = residual(fesolver,op,r_res,us_res)
  iA = get_matrix_index_map(op)
  ib = get_vector_index_map(op)
  return Snapshots(A,iA,r_jac),Snapshots(b,ib,r_res)
end

for n in (8,10,12,15)
  println("--------------------------------------------------------------------")
  println("TT algorithm, n = $(n)")
  domain = (0,1,0,1/3,0,1/6)
  partition = (n,ceil(n/3),ceil(n/6))
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
  rbsolver = RBSolver(fesolver,TTSVDReduction(fill(1e-4,5),energy_u(dΩ);nparams=50);nparams_test=10,nparams_res=30,nparams_jac=20)

  test_dir = datadir(joinpath("navier-stokes","3dcube_tensor_train_$(n)"))
  fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
  jacs_lin_u,ress_lin_u = _jacobian_and_residual(rbsolver,get_algebraic_operator(feop_lin_u),fesnaps[1])
  jacs_nlin_u,ress_nlin_u = _jacobian_and_residual(rbsolver,get_algebraic_operator(feop_nlin_u),fesnaps[1])

  for ϵ in (1e-1,1e-2,1e-3,1e-4,1e-5) #
    println("--------------------------------------------------------------------")
    println("TT algorithm, n = $(n), ϵ = $(ϵ)")

    tol = fill(ϵ,5)
    reduction = TTSVDReduction(tol,energy_u(dΩ);nparams=50)
    rbsolver = RBSolver(fesolver,reduction;nparams_test=10,nparams_res=30,nparams_jac=20)
    test_dir = datadir(joinpath("navier-stokes","3dcube_tensor_train_$(n)_$(ϵ)"))
    create_dir(test_dir)
    red_trial,red_test = reduced_fe_space(rbsolver,feop_u,fesnaps[1])
    op_lin = TransientPGOperator(get_algebraic_operator(feop_lin_u),red_trial,red_test)
    op_nlin = TransientPGOperator(get_algebraic_operator(feop_nlin_u),red_trial,red_test)
    rbops = map(zip((jacs_lin_u,jacs_nlin_u),(ress_lin_u,ress_nlin_u),(op_lin,op_nlin))) do (jacs,ress,op)
      red_lhs = reduced_jacobian(RBSteady.get_jacobian_reduction(rbsolver),op,jacs)
      red_rhs = reduced_residual(RBSteady.get_residual_reduction(rbsolver),op,ress)
      trians_rhs = get_domains(red_rhs)
      trians_lhs = map(get_domains,red_lhs)
      new_op = change_triangulation(op,trians_rhs,trians_lhs)
      TransientPGMDEIMOperator(new_op,red_lhs,red_rhs)
    end
    rbop = LinearNonlinearTransientPGMDEIMOperator(rbops...)
    rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps[1])
    results = rb_results(rbsolver,rbop,fesnaps[1],rbsnaps,rbstats,rbstats)

    save(test_dir,rbop)
    save(test_dir,results)

    println(results)
  end
end

n = 8
domain = (0,1,0,1/3,0,1/6)
partition = (n,ceil(n/3),ceil(n/6))
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
rbsolver = RBSolver(fesolver,TTSVDReduction(fill(1e-4,5),energy_u(dΩ);nparams=50);nparams_test=10,nparams_res=30,nparams_jac=20)

test_dir = datadir(joinpath("navier-stokes","3dcube_tensor_train_$(n)"))
fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
jacs_lin_u,ress_lin_u = _jacobian_and_residual(rbsolver,get_algebraic_operator(feop_lin_u),fesnaps[1])
jacs_nlin_u,ress_nlin_u = _jacobian_and_residual(rbsolver,get_algebraic_operator(feop_nlin_u),fesnaps[1])

ϵ = 1e-4

tol = fill(ϵ,5)
reduction = TTSVDReduction(tol,energy_u(dΩ);nparams=50)
rbsolver = RBSolver(fesolver,reduction;nparams_test=10,nparams_res=30,nparams_jac=20)
test_dir = datadir(joinpath("navier-stokes","3dcube_tensor_train_$(n)_$(ϵ)"))
create_dir(test_dir)
red_trial,red_test = reduced_fe_space(rbsolver,feop_u,fesnaps[1])
op_lin = TransientPGOperator(get_algebraic_operator(feop_lin_u),red_trial,red_test)
op_nlin = TransientPGOperator(get_algebraic_operator(feop_nlin_u),red_trial,red_test)
rbops = map(zip((jacs_lin_u,jacs_nlin_u),(ress_lin_u,ress_nlin_u),(op_lin,op_nlin))) do (jacs,ress,op)
  red_lhs = reduced_jacobian(RBSteady.get_jacobian_reduction(rbsolver),op,jacs)
  red_rhs = reduced_residual(RBSteady.get_residual_reduction(rbsolver),op,ress)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  TransientPGMDEIMOperator(new_op,red_lhs,red_rhs)
end
rbop = LinearNonlinearTransientPGMDEIMOperator(rbops...)
rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps[1])
results = rb_results(rbsolver,rbop,fesnaps[1],rbsnaps,rbstats,rbstats)

rbc1 = get_component(rbsnaps,1)
fec1 = get_component(son,1)
rbc2 = get_component(rbsnaps,2)
fec2 = get_component(son,2)
rbc3 = get_component(rbsnaps,3)
fec3 = get_component(son,3)
X = assemble_matrix(feop_lin_u,energy_u(dΩ))

compute_relative_error(fec1,rbc1,X)
compute_relative_error(fec2,rbc2,X)
compute_relative_error(fec3,rbc3,X)
compute_relative_error(son,rbsnaps,X)
