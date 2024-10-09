using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Visualization
using Gridap.ODEs
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

θ = 1
dt = 0.0025
t0 = 0.0
tf = 60*dt

pranges = [[1,9]*1e10,[0.25,0.42],[-4,4]*1e5,[-4,4]*1e5,[-4,4]*1e5]
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

domain = (0,1,0,1/3,0,1/3)

order = 2
degree = 2*order

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

energy(du,v,dΩ) = ∫(v⋅du)dΩ + ∫(∇(v)⊙∇(du))dΩ
energy(dΩ) = (du,v) -> energy(du,v,dΩ)

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

for n in (17,)#,17)
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

  fesolver = ThetaMethod(LUSolver(),dt,θ)

  tol = fill(1e-4,5)
  state_reduction = TTSVDReduction(tol,energy(dΩ);nparams=50)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res=30,nparams_jac=20,nparams_djac=1)
  test_dir = datadir(joinpath("test-elasticity","tt_$(n)"))
  create_dir(test_dir)

  r = realization(feop;nparams=60)
  fesnaps = load_snapshots(test_dir)
  # fesnaps,festats = solution_snapshots(rbsolver,feop,r,uh0μ)
  # save(test_dir,fesnaps)

  # println(festats)

  op = get_algebraic_operator(feop)
  jacs = jacobian_snapshots(rbsolver,op,fesnaps)
  newsave(test_dir,jacs;label="jac")
  ress = residual_snapshots(rbsolver,op,fesnaps)
  newsave(test_dir,ress;label="res")
end
