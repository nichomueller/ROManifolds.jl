using Gridap
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

energy_u(u,v) = ∫(v⋅u)dΩ + ∫(∇(v)⊙∇(u))dΩ

stiffness_u(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ
mass_u(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
res_u(μ,t,u,v,dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness_u(μ,t,u,v,dΩ) - ∫(v⋅hμt(μ,t))dΓn

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

  feop_u = TransientParamLinearFEOperator((stiffness_u,mass_u),res_u,ptspace,
      trial_u,test_u,trian_res,trian_stiffness,trian_mass)

  fesolver = ThetaMethod(LUSolver(),dt,θ)

  fesnaps = deserialize(RBSteady.get_snapshots_filename(test_dir))
  jacs,ress = _jacobian_and_residual(rbsolver,get_algebraic_operator(feop),fesnaps)

  for ϵ in (1e-1,1e-2,1e-3,1e-4,1e-5) #
    println("--------------------------------------------------------------------")
    println("TT algorithm, n = $(n), ϵ = $(ϵ)")

    tol = fill(ϵ,5)
    reduction = TTSVDReduction(tol,energy(dΩ);nparams=50)
    rbsolver = RBSolver(odesolver,reduction;nparams_test=10,nparams_res=30,nparams_jac=20)
    test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","3dcube_$(n)")))
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

    println(results)
  end
end
