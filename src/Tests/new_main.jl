begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")

  mesh = "cube2x2.json"
  test_path = "$root/tests/poisson/unsteady/$mesh"
  bnd_info = Dict("dirichlet" => [1,2,3,4,5,7,8],"neumann" => [6])
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

  res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u) + aμt(μ,t)*∇(v)⋅∇(u) - fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
  jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
  jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

  reffe = ReferenceFE(lagrangian,Float,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = PTTrialFESpace(test,g)
  feop = PTAffineFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tF,dt,θ = 0.,0.05,0.005,1
  uh0(μ) = interpolate_everywhere(u0(μ),trial(μ,t0))
  fesolver = ThetaMethod(LUSolver(),dt,θ)

  ϵ = 1e-4
  save_structures = true
  load_structures = true
  energy_norm = l2Norm()
  nsnaps_state = 80
  nsnaps_system = 30
  st_mdeim = false
  info = RBInfo(test_path;ϵ,load_structures,save_structures,energy_norm,
                nsnaps_state,nsnaps_system,st_mdeim)
  rbsolver = Backslash()
end

#OK
nsnaps = 10
p = realization(feop,nsnaps)


#TRY
# snap = collect_solutions(feop,fesolver,p;nsnaps)
# save(info,(snap,p))
snap,p = load(Snapshots,info),load(Table,info)
rbspace = compress_snapshots(info,snap,feop,fesolver,p)

rb_res = collect_compress_residual(info,feop,fesolver,rbspace,snap,p)
online_res = collect_residual_contributions(info,feop,fesolver,rb_res)

rb_jacs = collect_compress_jacobians(info,feop,fesolver,rbspace,snap,p)
online_jac = collect_jacobians_contributions(info,feop,fesolver,rb_jac)

compress_function(f,fesolver,Ω,p)
