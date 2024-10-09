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
θ = 1
dt = 0.0025
t0 = 0.0
tf = 60*dt

# parametric space
pranges = [[1,9]*1e10,[0.25,0.42],[-4,4]*1e5,[-4,4]*1e5,[-4,4]*1e5]
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1/3,0,1/3)

order = 2
degree = 2*order

# weak formulation

λ(μ) = μ[1]*μ[2]/((1+μ[2])*(1-2*μ[2]))
p(μ) = μ[1]/(2(1+μ[2]))

σ(ε,μ,t) = exp(sin(2*π*t/tf))*(λ(μ)*tr(ε)*one(ε) + 2*p(μ)*ε)
σ(μ,t) = ε -> σ(ε,μ,t)
σμt(μ,t) = TransientParamFunction(σ,μ,t)

h1(x,μ,t) = VectorValue(0.0,0.0,μ[3]*exp(sin(2*π*t/tf)))
h1(μ,t) = x->h1(x,μ,t)
h1μt(μ,t) = TransientParamFunction(h1,μ,t)

h2(x,μ,t) = VectorValue(0.0,μ[4]*exp(cos(2*π*t/tf)),0.0)
h2(μ,t) = x->h2(x,μ,t)
h2μt(μ,t) = TransientParamFunction(h2,μ,t)

h3(x,μ,t) = VectorValue(μ[5]*(1+t),0.0,0.0)
h3(μ,t) = x->h3(x,μ,t)
h3μt(μ,t) = TransientParamFunction(h3,μ,t)

g(x,μ,t) = VectorValue(0.0,0.0,0.0)
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫( ε(v) ⊙ (σμt(μ,t)∘ε(u)) )*dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,u,v,dΩ,dΓ1,dΓ2,dΓ3) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,u,v,dΩ) - (∫(v⋅h1μt(μ,t))dΓ1
  + ∫(v⋅h2μt(μ,t))dΓ2 + ∫(v⋅h3μt(μ,t))dΓ3)

energy(du,v,dΩ) =  ∫(v⋅du)dΩ + ∫(∇(v)⊙∇(du))dΩ
energy(dΩ) = (du,v) -> energy(du,v,dΩ)

energy_M(du,v,dΩ) =  ∫(v⋅du)dΩ
energy_M(dΩ) = (du,v) -> energy_M(du,v,dΩ)

fesolver = ThetaMethod(LUSolver(),dt,θ)

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

for n in (11,14,17)
  println("--------------------------------------------------------------------")
  println("TT algorithm, n = $(n)")

  partition = (n,floor(Int,n/3),floor(Int,n/3))
  model = TProductModel(domain,partition)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet",[1,3,5,7,13,15,17,19,25])
  add_tag_from_tags!(labels,"neumann1",[22])
  add_tag_from_tags!(labels,"neumann2",[24])
  add_tag_from_tags!(labels,"neumann3",[26])

  Ω = Triangulation(model)
  Γ1 = BoundaryTriangulation(model.model,tags="neumann1")
  Γ2 = BoundaryTriangulation(model.model,tags="neumann2")
  Γ3 = BoundaryTriangulation(model.model,tags="neumann3")

  dΩ = Measure(Ω,degree)
  dΓ1 = Measure(Γ1,degree)
  dΓ2 = Measure(Γ2,degree)
  dΓ3 = Measure(Γ3,degree)

  trian_res = (Ω.trian,Γ1,Γ2,Γ3)
  trian_stiffness = (Ω.trian,)
  trian_mass = (Ω.trian,)

  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
    trial,test,trian_res,trian_stiffness,trian_mass)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  test_dir = datadir(joinpath("test-elasticity","tt_$(n)"))

  fesnaps = load_snapshots(test_dir)
  jacs = newload(test_dir,feop.trian_jacs;label="jac")
  ress = newload(test_dir,feop.trian_res;label="res")

  x = select_snapshots(fesnaps,51:60)
  r = get_realization(x)

  for ϵ in (1e-2,1e-4)
    println("            ------------------------             ")
    println("TT algorithm, ϵ = $(ϵ)")
    test_dir_tol = joinpath(test_dir,"tol_$(ϵ)")
    create_dir(test_dir_tol)

    tol = fill(ϵ,5)
    # state_reduction = TTSVDReduction(tol,energy(dΩ);nparams=50)
    # rbsolver = RBSolver(fesolver,state_reduction;nparams_res=30,nparams_jac=20,nparams_djac=1)

    state_reduction_M = TTSVDReduction(tol,energy_M(dΩ);nparams=50)
    rbsolver = RBSolver(fesolver,state_reduction_M;nparams_res=30,nparams_jac=20,nparams_djac=1)

    red_test = RBSteady.load_fe_subspace(test_dir_tol,test;label="test_M")
    red_trial = red_test

    # red_test = RBSteady.load_fe_subspace(test_dir_tol,test;label="test")
    # red_trial = RBSteady.load_fe_subspace(test_dir_tol,trial;label="trial")
    odeop = get_algebraic_operator(feop)

    jac_red = RBSteady.get_jacobian_reduction(rbsolver)
    res_red = RBSteady.get_residual_reduction(rbsolver)
    red_lhs = reduced_jacobian(jac_red,red_trial,red_test,jacs)
    red_rhs = reduced_residual(res_red,red_test,ress)

    # trians_rhs = get_domains(red_rhs)
    # trians_lhs = map(get_domains,red_lhs)
    # new_odeop = change_triangulation(odeop,trians_rhs,trians_lhs)
    # rbop = GenericTransientRBOperator(new_odeop,red_trial,red_test,red_lhs,red_rhs)
    # # save(test_dir_tol,rbop)

    # rbop = load_operator(test_dir_tol,feop)

    # x̂,rbstats = solve(rbsolver,rbop,r)
    # x̂,rbstats = solve(rbsolver,rbop,r)

    # println(rbstats)

    # perf = rb_performance(rbsolver,rbop,x,x̂,rbstats,rbstats,r)

    # println(perf)
  end

  # println("--------------------------------------------------------------------")
  # println("Regular algorithm, n = $(n)")
  # test_dir′ = datadir(joinpath("test-elasticity","regular_$(n)"))

  # fesnaps′ = change_index_map(TrivialIndexMap,fesnaps)

  # test′ = TestFESpace(model.model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  # trial′ = TransientTrialParamFESpace(test′,gμt)
  # feop′ = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
  #   trial′,test′,trian_res,trian_stiffness,trian_mass)

  # x′ = select_snapshots(fesnaps′,51:60)
  # for ϵ in (1e-2,1e-3,1e-4)
  #   println("            ------------------------             ")
  #   println("Regular algorithm, ϵ = $(ϵ)")

  #   test_dir_tol′ = joinpath(test_dir′,"tol_$(ϵ)")
  #   create_dir(test_dir_tol′)

  #   state_reduction′ = TransientReduction(ϵ,energy(dΩ.measure);nparams=50)
  #   rbsolver′ = RBSolver(fesolver,state_reduction′;nparams_res=30,nparams_jac=20,nparams_djac=1)

  #   red_trial′,red_test′ = reduced_fe_space(rbsolver′,feop′,fesnaps′)
  #   # odeop′ = get_algebraic_operator(feop′)

  #   # ress′ = contribution(get_domains(ress)) do trian
  #   #   change_index_map(TrivialIndexMap,ress[trian])
  #   # end

  #   # mat_map = feop′.op.index_map.matrix_map
  #   # jacs′ = ()
  #   # for jtt in jacs
  #   #   j = contribution(get_domains(jtt)) do trian
  #   #     s = jtt[trian]
  #   #     alls = jtt[trian].snaps
  #   #     allv = get_values(alls)
  #   #     allr = get_realization(alls)
  #   #     select_snapshots(Snapshots(allv,mat_map,allr),s.prange)
  #   #   end
  #   #   jacs′ = (jacs′...,j)
  #   # end

  #   # jac_red′ = RBSteady.get_jacobian_reduction(rbsolver′)
  #   # res_red′ = RBSteady.get_residual_reduction(rbsolver′)
  #   # red_lhs′ = reduced_jacobian(jac_red′,red_trial′,red_test′,jacs′)
  #   # red_rhs′ = reduced_residual(res_red′,red_test′,ress′)

  #   # trians_rhs′ = get_domains(red_rhs′)
  #   # trians_lhs′ = map(get_domains,red_lhs′)
  #   # new_odeop′ = change_triangulation(odeop′,trians_rhs′,trians_lhs′)
  #   # rbop′ = GenericTransientRBOperator(new_odeop′,red_trial′,red_test′,red_lhs′,red_rhs′)
  #   # save(test_dir_tol′,rbop′)

  #   rbop′ = load_operator(test_dir_tol′,feop′)

  #   x̂′,rbstats′ = solve(rbsolver′,rbop′,r)
  #   x̂′,rbstats′ = solve(rbsolver′,rbop′,r)
  #   println(rbstats′)

  #   perf′ = rb_performance(rbsolver′,rbop′,x′,x̂′,rbstats′,rbstats′,r)
  #   println(perf′)
  # end
end

# n = 14
# partition = (n,floor(Int,n/3),floor(Int,n/3))
# model = TProductModel(domain,partition)
# labels = get_face_labeling(model)
# add_tag_from_tags!(labels,"dirichlet",[1,3,5,7,13,15,17,19,25])
# add_tag_from_tags!(labels,"neumann1",[22])
# add_tag_from_tags!(labels,"neumann2",[24])
# add_tag_from_tags!(labels,"neumann3",[26])

# Ω = Triangulation(model)
# Γ1 = BoundaryTriangulation(model.model,tags="neumann1")
# Γ2 = BoundaryTriangulation(model.model,tags="neumann2")
# Γ3 = BoundaryTriangulation(model.model,tags="neumann3")

# dΩ = Measure(Ω,degree)
# dΓ1 = Measure(Γ1,degree)
# dΓ2 = Measure(Γ2,degree)
# dΓ3 = Measure(Γ3,degree)

# trian_res = (Ω.trian,Γ1,Γ2,Γ3)
# trian_stiffness = (Ω.trian,)
# trian_mass = (Ω.trian,)

# reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
# test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
# trial = TransientTrialParamFESpace(test,gμt)
# feop = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
#   trial,test,trian_res,trian_stiffness,trian_mass)
# uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

# test_dir = datadir(joinpath("test-elasticity","tt_$(n)"))

# fesnaps = load_snapshots(test_dir)

# X = assemble_matrix(feop,energy(dΩ))
# S = select_snapshots(fesnaps,1:50)
# Φ1 = projection(state_reduction,S,X[1])
# Φ2 = projection(state_reduction,S,X[2])
# Φ3 = projection(state_reduction,S,X[3])
# ΦB = RBSteady.block_cores(Φ1.cores[1:4],Φ2.cores[1:4],Φ3.cores[1:4])
# rr = state_reduction.red_style
# ΦB1,R1 = RBSteady.reduce_rank(rr[1],ΦB[1])
# ΦB2,R2 = RBSteady.reduce_rank(rr[2],RBSteady.absorb(ΦB[2],R1))
# ΦB3,R3 = RBSteady.reduce_rank(rr[3],RBSteady.absorb(ΦB[3],R2))
# STOP
# x = select_snapshots(fesnaps,51:60)
# r = get_realization(x)

# ϵ = 1e-5
# test_dir_tol = joinpath(test_dir,"tol_$(ϵ)")
# rbop = load_operator(test_dir_tol,feop)

# tol = fill(ϵ,5)
# state_reduction = TTSVDReduction(tol,energy(dΩ);nparams=50)
# rbsolver = RBSolver(fesolver,state_reduction;nparams_res=30,nparams_jac=20,nparams_djac=1)

# x̂,rbstats = solve(rbsolver,rbop,r)

# xrb = inv_project(get_trial(rbop)(r),x̂)
# rbsnaps = Snapshots(xrb,get_vector_index_map(rbop),r)

# sol,sol_approx = x,rbsnaps
# X = assemble_matrix(feop,energy(dΩ))
# compute_relative_error(sol,sol_approx,X)


# ################################################################################
# test_dir′ = datadir(joinpath("test-elasticity","regular_$(n)"))

# fesnaps′ = change_index_map(TrivialIndexMap,fesnaps)

# test′ = TestFESpace(model.model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
# trial′ = TransientTrialParamFESpace(test′,gμt)
# feop′ = TransientParamLinearFEOperator((stiffness,mass),res,ptspace,
#   trial′,test′,trian_res,trian_stiffness,trian_mass)

# x′ = select_snapshots(fesnaps′,51:60)

# test_dir_tol′ = joinpath(test_dir′,"tol_$(ϵ)")

# state_reduction′ = TransientReduction(ϵ,energy(dΩ.measure);nparams=50)
# rbsolver′ = RBSolver(fesolver,state_reduction′;nparams_res=30,nparams_jac=20,nparams_djac=1)

# rbop′ = load_operator(test_dir_tol′,feop′)

# x̂′,rbstats′ = solve(rbsolver′,rbop′,r)

# perf′ = rb_performance(rbsolver′,rbop′,x′,x̂′,rbstats′,rbstats′,r)

# xrb′ = inv_project(get_trial(rbop′)(r),x̂′)
# rbsnaps′ = Snapshots(xrb′,get_vector_index_map(rbop′),r)

for n in (11,14,17)
  test_dir = datadir(joinpath("test-elasticity","tt_$(n)"))
  for ϵ in (1e-2,1e-4)
    test_dir_tol = joinpath(test_dir,"tol_$(ϵ)")
    basis = RBSteady.load_projection(test_dir_tol;label="test")
    basis′ = RBSteady.load_projection(test_dir_tol;label="test_M")
    println("ndofs H1: $(size.(basis.cores))")
    println("ndofs L2: $(size.(basis′.cores))")
  end
end
