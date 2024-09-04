using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.CellData
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

# time marching
θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 0.15

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

# weak formulation
a(x,μ,t) = 1+exp(-sin(t)^2*x[1]/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(x,μ,t) = abs.(sin(9*pi*t/(5*μ[3])))
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

g(x,μ,t) = exp(-x[1]/μ[2])*abs(1-cos(9*pi*t/5)+sin(9*pi*t/(5*μ[3])))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = 0
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ) = ∫(fμt(μ,t)*v)dΩ
res(μ,t,u,v,dΩ) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ)

induced_norm(du,v,dΩ) = ∫(∇(v)⋅∇(du))dΩ
induced_norm(dΩ) = (du,v) -> ∫(∇(v)⋅∇(du))dΩ

order = 2
degree = 2*order

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

for n in (15,)#(8,10,12,15)
  domain = (0,1,0,1,0,1)
  partition = (n,n,n)
  model = TProductModel(domain,partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet","boundary")

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  trian_res = (Ω.trian,)
  trian_stiffness = (Ω.trian,)
  trian_mass = (Ω.trian,)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm(dΩ),ptspace,
    trial,test,trian_res,trian_stiffness,trian_mass)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  fesolver = ThetaMethod(LUSolver(),dt,θ)
  rbsolver = RBSolver(fesolver,1e-4;nsnaps_state=50,nsnaps_test=10,nsnaps_res=30,nsnaps_jac=20)
  test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","3dcube_tensor_train_$(n)")))
  fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
  jacs,ress = _jacobian_and_residual(rbsolver,get_algebraic_operator(feop),fesnaps)

  for ϵ in (1e-1,1e-2,1e-3,1e-4,1e-5)
    println("--------------------------------------------------------------------")
    println("TT algorithm, n = $(n), ϵ = $(ϵ)")

    rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_res=30,nsnaps_jac=20)
    test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","3dcube_tensor_train_$(n)")))
    red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
    op = TransientPGOperator(get_algebraic_operator(feop),red_trial,red_test)
    red_lhs = reduced_jacobian(rbsolver,op,jacs)
    red_rhs = reduced_residual(rbsolver,op,ress)
    trians_rhs = get_domains(red_rhs)
    trians_lhs = map(get_domains,red_lhs)
    new_op = change_triangulation(op,trians_rhs,trians_lhs)
    rbop = TransientPGMDEIMOperator(new_op,red_lhs,red_rhs)

    rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
    results = rb_results(rbsolver,rbop,fesnaps,rbsnaps)

    save(test_dir,rbop)
    save(test_dir,results)

    println(compute_error(results))
    println(get_timer(results))
  end
end
