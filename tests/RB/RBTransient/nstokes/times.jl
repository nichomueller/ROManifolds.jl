using Gridap
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using Test
using DrWatson
using Serialization
using BenchmarkTools

using ROM
using ROM.ParamDataStructures
using ROM.ParamSteady
using ROM.ParamFESpaces

# time marching
θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 60*dt

# parametric space
pranges = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pranges,tdomain)

order = 2
degree = 2*order+1

const Re′ = 100.0
a(x,μ,t) = μ[1]/Re′
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

const W = 0.5
inflow(μ,t) = abs(1-cos(2π*t/tf)+μ[3]*sin(μ[2]*2π*t/tf)/100)
g_in(x,μ,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(x,μ,t) = VectorValue(0.0,0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = TransientParamFunction(g_0,μ,t)

u0(x,μ) = VectorValue(0.0,0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

# loop on mesh
for h in ("h007","h005","h0035")
  model_dir = datadir(joinpath("models","model_circle_$h.json"))
  model = DiscreteModelFromFile(model_dir)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
  add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
  add_tag_from_tags!(labels,"dirichlet",["inlet"])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
  dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

  djac(μ,t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
  jac(μ,t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(aμt(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
  res(μ,t,(u,p),(v,q)) = c(u,v) + ∫(v⋅∂t(u))dΩ + ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

  reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,
    dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
    dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
  trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_in,gμt_0])
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(model,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = TransientParamFEOperator(res,(jac,djac),ptspace,trial,test)

  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  odeop = get_algebraic_operator(feop)
  ws = (1,1)
  us(x) = (x,x)

  # loop on params
  for nparams in 1:10
    r = realization(ptspace;nparams)

    U = trial(r)
    x = get_free_dof_values(zero(U))

    paramcache = allocate_paramcache(odeop,r,(x,x))
    stageop = ParamStageOperator(odeop,paramcache,r,us,ws)

    println("Residual time with h = $h, nparams = $nparams:")
    b = allocate_residual(stageop,x)
    @btime residual!($b,$stageop,$x)

    println("Jacobian time with h = $h, nparams = $nparams:")
    A = allocate_jacobian(stageop,x)
    @btime jacobian!($A,$stageop,$x)

    # println("Solve time with h = $h, nparams = $nparams:")
    # @btime solve!(x,LUSolver(),A,b)
  end
end

# Gridap
μ = rand(3)
g′_in(x,t) = VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0,0.0)
g′_in(t) = x->g′_in(x,t)
g′_0(x,t) = VectorValue(0.0,0.0,0.0)
g′_0(t) = x->g′_0(x,t)

# loop on mesh
for h in ("h007","h005","h0035")
  model_dir = datadir(joinpath("models","model_circle_$h.json"))
  model = DiscreteModelFromFile(model_dir)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet_noslip",["cylinders","walls"])
  add_tag_from_tags!(labels,"dirichlet_nopenetration",["top_bottom"])
  add_tag_from_tags!(labels,"dirichlet",["inlet"])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
  dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

  djac(t,(uₜ,pₜ),(dut,dpt),(v,q)) = ∫(v⋅dut)dΩ
  jac(t,(u,p),(du,dp),(v,q)) = dc(u,du,v) + ∫(a(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
  res(t,(u,p),(v,q)) = c(u,v) + ∫(v⋅u)dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

  reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,
    dirichlet_tags=["inlet","dirichlet_nopenetration","dirichlet_noslip"],
    dirichlet_masks=[[true,true,true],[false,false,true],[true,true,true]])
  trial_u = TransientTrialFESpace(test_u,[g′_in,g′_in,g′_0])
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(model,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = TransientMultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = TransientMultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = TransientFEOperator(res,(jac,djac),trial,test)

  xh0 = interpolate_everywhere([u0(μ),p0(μ)],trial(t0))

  odeop = get_algebraic_operator(feop)
  ws = (1,1)
  us(x) = (x,x)

  U = trial(t0)
  x = get_free_dof_values(zero(U))

  odeopcache = allocate_odeopcache(odeop,t0,(x,x))
  stageop = NonlinearStageOperator(odeop,odeopcache,t0,us,ws)

  println("Gridap residual time with h = $h:")
  b = allocate_residual(stageop,x)
  @btime residual!($b,$stageop,$x)

  println("Gridap Jacobian time with h = $h:")
  A = allocate_jacobian(stageop,x)
  @btime jacobian!($A,$stageop,$x)

  # println("Solve time with h = $h:")
  # @btime solve!(x,LUSolver(),A,b)
end
