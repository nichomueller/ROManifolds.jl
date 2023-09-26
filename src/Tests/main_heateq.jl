begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")

  mesh = "cube2x2.json"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  # mesh = "elasticity_3cyl2D.json"
  # bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
  test_path = "$root/tests/poisson/unsteady/$mesh"
  order = 1
  degree = 2

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ranges = fill([1.,2.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  f(x,μ,t) = 1.
  f(μ,t) = x->f(x,μ,t)
  fμt(μ,t) = PTFunction(f,μ,t)

  h(x,μ,t) = abs(cos(t/μ[3]))
  h(μ,t) = x->h(x,μ,t)
  hμt(μ,t) = PTFunction(h,μ,t)

  g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = 0
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)

  m(μ,t,dut,v) = ∫ₚ(v*dut,dΩ)
  lhs(μ,t,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
  rhs(μ,t,v) = ∫ₚ(fμt(μ,t)*v,dΩ) + ∫ₚ(hμt(μ,t)*v,dΓn)

  reffe = ReferenceFE(lagrangian,Float,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = PTTrialFESpace(test,g)
  feop = PTAffineFEOperator(m,lhs,rhs,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.05,0.005,1
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  save_structures = true
  load_structures = true
  energy_norm = l2Norm()
  nsnaps_state = 80
  nsnaps_system = 30
  st_mdeim = false
  info = RBInfo(test_path;ϵ,load_structures,save_structures,energy_norm,
                nsnaps_state,nsnaps_system,st_mdeim)
end
