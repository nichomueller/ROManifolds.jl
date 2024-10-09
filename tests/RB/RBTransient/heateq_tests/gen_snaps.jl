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

function newsave(dir,contrib::ArrayContribution;label::String="")
  for (i,c) in enumerate(get_values(contrib))
    l = RBSteady._get_label(label,i)
    save(dir,c;label=l)
  end
end

function newsave(dir,contrib::TupOfArrayContribution;label::String="")
  for (i,c) in enumerate(contrib)
    l = RBSteady._get_label(label,i)
    newsave(dir,c;label=l)
  end
end

for n in (12,)#(8,10,12)
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

  tol = fill(1e-4,4)
  state_reduction = TTSVDReduction(tol,energy(dΩ);nparams=50)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res=30,nparams_jac=20,nparams_djac=1)
  test_dir = datadir(joinpath("test-heateq","tt_$(n)"))
  create_dir(test_dir)

  # r = realization(feop;nparams=60)
  # fesnaps,festats = solution_snapshots(rbsolver,feop,r,uh0μ)
  # save(test_dir,fesnaps)
  fesnaps = load_snapshots(test_dir)

  # println(festats)

  op = get_algebraic_operator(feop)
  jacs = jacobian_snapshots(rbsolver,op,fesnaps)
  # ress = residual_snapshots(rbsolver,op,fesnaps)

  newsave(test_dir,jacs;label="jac")
  # newsave(test_dir,ress;label="res")
end
