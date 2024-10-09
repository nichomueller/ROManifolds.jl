using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Visualization
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
dt = 0.0025
t0 = 0.0
tf = 60*dt

pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

order = 2
degree = 2*order

a(x,μ,t) = 1+exp(sin(t)/sum(μ))#1+exp(-sin(2π*t/tf)^2*(1-x[2])/sum(μ))
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

g0(x,μ,t) = VectorValue(0.0,0.0,0.0)
g0(μ,t) = x->g0(x,μ,t)
g0μt(μ,t) = TransientParamFunction(g0,μ,t)

inflow(μ,t) = abs(μ[1]*cos(t)+μ[2]*sin(t))
h(x,μ,t) = -VectorValue(x[2]*(1-x[2])*x[3]*(1-x[3])*inflow(μ,t),0.0,0.0)
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

jac_lin(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
res_lin(μ,t,(u,p),(v,q),dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + jac_lin(μ,t,(u,p),(v,q),dΩ) - ∫(v⋅hμt(μ,t))dΓn
# res_lin(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + jac_lin(μ,t,(u,p),(v,q),dΩ)

# jac(μ,t,(u,p),(du,dp),(v,q),dΩ) = jac_lin(μ,t,(du,dp),(v,q),dΩ) + dc(u,du,v,dΩ)
# djac(μ,t,(uₜ,pₜ),(duₜ,dpₜ),(v,q),dΩ) = mass(μ,t,(duₜ,dpₜ),(v,q),dΩ)
# res(μ,t,(u,p),(v,q),dΩ,dΓn) = res_lin(μ,t,(u,p),(v,q),dΩ,dΓn) + c(u,v,dΩ)

energy_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ
energy_u(dΩ) = (u,v) -> energy_u(u,v,dΩ)

stiffness_u(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ
mass_u(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
res_u(μ,t,u,v,dΩ,dΓn) = ∫(v⋅∂t(u))dΩ + stiffness_u(μ,t,u,v,dΩ) - ∫(v⋅hμt(μ,t))dΓn

for n in (5,)#10,12,15)
  println("--------------------------------------------------------------------")
  println("TT algorithm, n = $(n)")
  domain = (0,1,0,1,0,1)
  partition = (n,n,n)
  model = TProductModel(domain,partition) #CartesianDiscreteModel(domain,partition)#
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet0",collect(1:24))
  add_tag_from_tags!(labels,"neumann",[25])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model.model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  trian_res = (Ω.trian,Γn)
  trian_jac = (Ω.trian,)
  trian_djac = (Ω.trian,)
  # trian_res = (Ω,)
  # trian_jac = (Ω,)
  # trian_djac = (Ω,)

  reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
  # test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["neumann","dirichlet0"])
  trial_u = TransientTrialParamFESpace(test_u,g0μt)
  # trial_u = TransientTrialParamFESpace(test_u,[hμt,g0μt])
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(Ω,reffe_p;conformity=:C0)
  # test_p = TestFESpace(model,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = TransientParamLinearFEOperator((jac_lin,mass),res_lin,ptspace,
    trial,test,trian_res,trian_jac,trian_djac)

  feop_u = TransientParamLinearFEOperator((stiffness_u,mass_u),res_u,ptspace,
    trial_u,test_u,trian_res,trian_jac,trian_djac)

  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  fesolver = ThetaMethod(LUSolver(),dt,θ)

  tol = fill(1e-4,5)
  reduction = TTSVDReduction(tol,energy_u(dΩ);nparams=50)
  rbsolver = RBSolver(fesolver,reduction;nparams_res=30,nparams_jac=20,nparams_djac=1)
  test_dir = datadir("test-nstokes")
  create_dir(test_dir)

  r = realization(feop;nparams=60)
  fesnaps,festats = solution_snapshots(rbsolver,feop,xh0μ;r)
  save(test_dir,fesnaps)

  println(festats)

  # s1 = select_snapshots(fesnaps[1],1)
  # s1 = change_index_map(TrivialIndexMap,s1)
  # r1 = get_realization(s1)
  # U1 = trial_u(r1)

  # dir = datadir("plts")
  # createpvd(dir) do pvd
  #   for i in param_eachindex(r1)
  #     file = dir*"/u$i"*".vtu"
  #     Ui = param_getindex(U1,i)
  #     vi = s1[:,i,1]
  #     uhi = FEFunction(Ui,vi)
  #     pvd[i] = createvtk(Ω.trian,file,cellfields=["u"=>uhi])
  #   end
  # end
  fesnaps_u = fesnaps[1]
  rbop = reduced_operator(rbsolver,feop_u,fesnaps_u)
  save(test_dir,rbop)
  ronline = r[51:60,:]
  x = select_snapshots(fesnaps_u,51:60)
  x̂,rbstats = solve(rbsolver,rbop,ronline)
  perf = rb_performance(rbsolver,rbop,x,x̂,rbstats,rbstats,ronline)
  println(perf)
end

red_style = TTSVDReduction(fill(1e-4,5),energy(dΩ);nparams=50)
soff = select_snapshots(fesnaps,1:10)
# basis = reduced_basis(red_style,feop,soff)

X = assemble_matrix(feop,energy(dΩ))
basis_u = reduced_basis(red_style,soff[1],X[1,1])
basis_p = reduced_basis(red_style,soff[2],X[2,2])

coupling((du,dp),(v,q)) = ∫(dp*(∂ₓ₁(v)+∂ₓ₂(v)+∂ₓ₃(v)))dΩ
B = assemble_matrix(feop,coupling)


STOP
# model′ = model.model
# Ω′ = Ω.trian
# dΩ′ = dΩ.measure

# test_u′ = TestFESpace(model′,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0"])
# trial_u′ = TransientTrialParamFESpace(test_u′,g0μt)
# test_p′ = TestFESpace(model′,reffe_p;conformity=:C0)
# trial_p′ = TrialFESpace(test_p′)

# reduction′ = TransientReduction(1e-4,energy_u(dΩ′);nparams=50)
# rbsolver′ = RBSolver(fesolver,reduction′;nparams_res=30,nparams_jac=20,nparams_djac=1)

# feop_u′ = TransientParamLinearFEOperator((stiffness_u,mass_u),res_u,ptspace,
#   trial_u′,test_u′,trian_res,trian_jac,trian_djac)

# fesnaps_u′ = change_index_map(TrivialIndexMap,fesnaps_u)
# rbop′ = reduced_operator(rbsolver′,feop_u′,fesnaps_u′)
# x′ = select_snapshots(fesnaps_u′,51:60)
# x̂,rbstats = solve(rbsolver,rbop,ronline)
# perf = rb_performance(rbsolver,rbop,x,x̂,rbstats,rbstats,ronline)
