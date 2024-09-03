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
# a(x,μ,t) = μ[1]*(1+exp(-sin(t)^2/sum(μ)))
# a(μ,t) = x->a(x,μ,t)
# aμt(μ,t) = TransientParamFunction(a,μ,t)

# f(x,μ,t) = abs(1-cos(9*π*t/(5*tf)))
# f(μ,t) = x->f(x,μ,t)
# fμt(μ,t) = TransientParamFunction(f,μ,t)

# g(x,μ,t) = exp(-x[1]/μ[2])*abs(sin(9*π*t/(5*tf*μ[3])))
# g(μ,t) = x->g(x,μ,t)
# gμt(μ,t) = TransientParamFunction(g,μ,t)

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

for I in (:AbstractIndexMap,:(AbstractArray{<:AbstractIndexMap}))
  @eval begin
    function _change_index_map(f,a::ArrayContribution)
      contribution(a.trians) do trian
        change_index_map(f,a[trian])
      end
    end

    function _change_index_map(f,a::TupOfArrayContribution)
      map(a->_change_index_map(f,a),a)
    end
  end
end

for n in (15,) #8,10,12,
  # geometry
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
  temp_rbsolver = RBSolver(fesolver,1e-4;nsnaps_state=50,nsnaps_test=10,nsnaps_res=30,nsnaps_jac=20)
  fesnaps = fe_solutions(temp_rbsolver,feop,uh0μ)
  op = get_algebraic_operator(feop)
  jacs,ress = _jacobian_and_residual(temp_rbsolver,op,fesnaps)

  _test = TestFESpace(model.model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  _trial = TransientTrialParamFESpace(_test,gμt)
  _feop = TransientParamLinearFEOperator((stiffness,mass),res,induced_norm(dΩ.measure),ptspace,
    _trial,_test,trian_res,trian_stiffness,trian_mass)
  _uh0μ(μ) = interpolate_everywhere(u0μ(μ),_trial(μ,t0))
  _fesnaps = change_index_map(TrivialIndexMap,fesnaps)
  _op = get_algebraic_operator(_feop)
  _jacs = _change_index_map(TrivialIndexMap,jacs)
  _ress = _change_index_map(TrivialIndexMap,ress)

  for ϵ in (1e-1,1e-2,1e-3,1e-4,1e-5)
    println("--------------------------------------------------------------------")
    println("TT algorithm, n = $(n), ϵ = $(ϵ)")

    rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_res=30,nsnaps_jac=20)
    test_dir = get_test_directory(rbsolver,dir=datadir(joinpath("heateq","3dcube_tensor_train_$(n)_$(ϵ)")))
    save(test_dir,fesnaps)

    # rbop = reduced_operator(rbsolver,feop,fesnaps)
    red_trial,red_test = reduced_fe_space(rbsolver,feop,fesnaps)
    op = TransientPGOperator(get_algebraic_operator(feop),red_trial,red_test)
    red_lhs = reduced_jacobian(rbsolver,op,jacs)
    red_rhs = reduced_residual(rbsolver,op,ress)
    trians_rhs = get_domains(red_rhs)
    trians_lhs = map(get_domains,red_lhs)
    new_op = change_triangulation(op,trians_rhs,trians_lhs)
    rbop = TransientPGMDEIMOperator(new_op,red_lhs,red_rhs)

    rbsnaps,cache = solve(rbsolver,rbop,fesnaps)
    results = rb_results(rbsolver,rbop,fesnaps,rbsnaps)

    println(compute_error(results))
    println(get_timer(results))

    save(test_dir,rbop)
    save(test_dir,results)

    # regular
    println("--------------------------------------------------------------------")
    println("Regular algorithm, n = $(n), ϵ = $(ϵ)")

    _rbsolver = RBSolver(fesolver,ϵ;nsnaps_state=50,nsnaps_test=10,nsnaps_res=30,nsnaps_jac=20)
    _test_dir = get_test_directory(_rbsolver,dir=datadir(joinpath("heateq","3dcube_$(n)_$(ϵ)")))
    save(_test_dir,_fesnaps)
    # _rbop = reduced_operator(_rbsolver,_feop,_fesnaps)
    _red_trial,_red_test = reduced_fe_space(_rbsolver,feop,_fesnaps)
    _op = TransientPGOperator(get_algebraic_operator(_feop),_red_trial,_red_test)
    _red_lhs = reduced_jacobian(_rbsolver,_op,_jacs)
    _red_rhs = reduced_residual(_rbsolver,_op,_ress)
    _trians_rhs = get_domains(_red_rhs)
    _trians_lhs = map(get_domains,_red_lhs)
    _new_op = change_triangulation(_op,_trians_rhs,_trians_lhs)
    _rbop = TransientPGMDEIMOperator(_new_op,_red_lhs,_red_rhs)

    _rbsnaps,_cache = solve(_rbsolver,_rbop,_fesnaps)
    _results = rb_results(_rbsolver,_rbop,_fesnaps,_rbsnaps)

    println(compute_error(_results))
    println(get_timer(_results))

    save(_test_dir,_rbop)
    save(_test_dir,_results)
  end
end

# s1 = select_snapshots(fesnaps,1)
# _s1 = change_index_map(TrivialIndexMap,s1)
# # sa1 = select_snapshots(results.sol_approx[1],1)
# # e1 = s1 - sa1
# r1 = get_realization(s1)
# U1 = trial(r1)

# # using Gridap.Visualization
# # dir = datadir("plts")
# # createpvd(dir) do pvd
# #   for i in param_eachindex(r1)
# #     file = dir*"/u$i"*".vtu"
# #     Ui = param_getindex(U1,i)
# #     # vi = e1[:,i,1]
# #     vi = _s1[:,i,1]
# #     uhi = FEFunction(Ui,vi)
# #     pvd[i] = createvtk(Ω.trian,file,cellfields=["u"=>uhi])
# #   end
# # end
