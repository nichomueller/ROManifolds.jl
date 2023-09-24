begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")

  mesh = "cube2x2.json"
  test_path = "$root/tests/navier-stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
  order = 2
  degree = 4

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  pspace = PSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)
  aμt(μ,t) = PTFunction(a,μ,t)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)
  p0μ(μ) = PFunction(p0,μ)

  m(μ,t,u,v) = ∫ₚ(v⋅u,dΩ)
  a(μ,t,(u,p),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(u)),dΩ)
  c(μ,t,u,v) = ∫ₚ(v⊙(conv∘(u,∇(u))),dΩ)
  dc(μ,t,u,du,v) = ∫ₚ(v⊙(dconv∘(du,∇(du),u,∇(u))),dΩ)

  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = m(μ,t,dut,v)
  jac(μ,t,(u,p),(du,dp),(v,q)) = a(μ,t,(du,dp),(v,q)) + dc(μ,t,u,du,v)
  res(μ,t,(u,p),(v,q)) = a(μ,t,(u,p),(v,q)) + c(μ,t,u,v)

  reffe_u = Gridap.ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = Gridap.ReferenceFE(lagrangian,Float,order-1)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial_u = PTTrialFESpace(test_u,g)
  test_p = TestFESpace(model,reffe_p;conformity=:H1,constraint=:zeromean)
  trial_p = TrialFESpace(test_p)
  test = PTMultiFieldFESpace([test_u,test_p])
  trial = PTMultiFieldFESpace([trial_u,trial_p])
  feop = PTFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.05,0.005,0.5
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
  ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
  xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))

  nls = NewtonRaphsonSolver(LUSolver(),1e-10,20)
  fesolver = ThetaMethod(nls,dt,θ)
end
