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

energy(du,v,dΩ) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
energy(dΩ) = (du,v) -> energy(du,v,dΩ)

order = 2
degree = 2*order

function newload(dir,trian::Tuple{Vararg{Triangulation}};label="")
  a = ()
  for (i,t) in enumerate(trian)
    l = RBSteady._get_label(label,i)
    ai = load_snapshots(dir;label=l)
    a = (a...,ai)
  end
  return Contribution(a,trian)
end

function newload(dir,trian::Tuple{Vararg{Tuple{Vararg{Triangulation}}}};label="")
  a = ()
  for (i,t) in enumerate(trian)
    l = RBSteady._get_label(label,i)
    a = (a...,newload(dir,t;label=l))
  end
  return a
end

for n in (8,10,12)
  println("--------------------------------------------------------------------")
  println("TT algorithm, n = $(n)")
  domain = (0,1,0,1,0,1)
  partition = (n,n,n)
  model = TProductModel(domain,partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet",setdiff(1:26,[23,24]))

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  trian_res = (Ω.trian,)
  trian_stiffness = (Ω.trian,)
  trian_mass = (Ω.trian,)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
    trial,test,trian_res,trian_stiffness,trian_mass)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  fesolver = ThetaMethod(LUSolver(),dt,θ)

  test_dir = datadir(joinpath("test-heateq","tt_$(n)"))

  fesnaps = load_snapshots(test_dir)
  jacs = newload(test_dir,feop.trian_jacs;label="jac")
  ress = newload(test_dir,feop.trian_res;label="res")

  # x = select_snapshots(fesnaps,51:60)
  # r = get_realization(x)

  for ϵ in (1e-4,)
    println("            ------------------------             ")
    println("TT algorithm, ϵ = $(ϵ)")
    test_dir_tol = joinpath(test_dir,"tol_$(ϵ)")
    create_dir(test_dir_tol)

    tol = fill(ϵ,4)
    state_reduction = TTSVDReduction(tol,energy(dΩ);nparams=50)
    rbsolver = RBSolver(fesolver,state_reduction;nparams_res=30,nparams_jac=20)

    # red_test,red_trial = reduced_fe_space(rbsolver,feop,fesnaps)
    red_test = RBSteady.load_fe_subspace(test_dir_tol,test;label="test")
    red_trial = RBSteady.load_fe_subspace(test_dir_tol,trial;label="trial")
    odeop = get_algebraic_operator(feop)

    jac_red = RBSteady.get_jacobian_reduction(rbsolver)
    res_red = RBSteady.get_residual_reduction(rbsolver)
    red_lhs = reduced_jacobian(jac_red,red_trial,red_test,jacs)
    red_rhs = reduced_residual(res_red,red_test,ress)

    # trians_rhs = get_domains(red_rhs)
    # trians_lhs = map(get_domains,red_lhs)
    # new_odeop = change_triangulation(odeop,trians_rhs,trians_lhs)
    # rbop = GenericTransientRBOperator(new_odeop,red_test,red_trial,red_lhs,red_rhs)
    # # save(test_dir_tol,rbop)

    # rbop = load_operator(test_dir_tol,feop)

    # x̂,rbstats = solve(rbsolver,rbop,r)
    # # x̂,rbstats = solve(rbsolver,rbop,r)

    # # println(rbstats)

    # perf = rb_performance(rbsolver,rbop,x,x̂,rbstats,rbstats,r)

    # println(perf)
  end
end

# for n in (8,10,12)
#   test_dir = datadir(joinpath("test-heateq","tt_$(n)"))
#   for ϵ in (1e-1,1e-2,1e-3,1e-4,1e-5)
#     test_dir_tol = joinpath(test_dir,"tol_$(ϵ)")
#     basis = RBSteady.load_projection(test_dir_tol;label="test")
#     r = size.(basis.cores)
#     println("n = $n, tol = $ϵ: $r")
#   end
# end
