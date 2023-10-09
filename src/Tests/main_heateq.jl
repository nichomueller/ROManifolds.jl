begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")

  mesh = "elasticity_3cyl2D.json"
  bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
  test_path = "$root/tests/poisson/unsteady/$mesh"
  order = 1
  degree = 2

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ranges = fill([1.,10.],3)
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

  jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)
  jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
  res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⋅∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)

  reffe = ReferenceFE(lagrangian,Float,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = PTTrialFESpace(test,g)
  feop = PTAffineFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.05,0.005,0.5
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = true
  save_solutions = true
  load_structures = false
  save_structures = true
  energy_norm = :l2
  nsnaps_state = 50
  nsnaps_system = 20
  nsnaps_test = 10
  st_mdeim = false
  info = RBInfo(test_path;ϵ,load_solutions,save_solutions,load_structures,save_structures,
                energy_norm,nsnaps_state,nsnaps_system,nsnaps_test,st_mdeim)
  reduced_basis_model(info,feop,fesolver)
end

sols,params = load(info,(AbstractSnapshots,Table))
rbspace = load(info,AbstractRBSpace)
rbrhs,rblhs = load(info,(AbstractRBAlgebraicContribution,Vector{AbstractRBAlgebraicContribution}))
snaps_test,params_test = load_test(info,feop,fesolver)

println("Solving linear RB problems\n")
x = initial_guess(sols,params,params_test)
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,rbspace,snaps_test,params_test)
rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,x,params_test)
lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,x,params_test)

stats = @timed begin
  rb_snaps_test = rb_solve(fesolver.nls,rhs,lhs)
end
approx_snaps_test = recast(rbspace,rb_snaps_test)
# approx_snaps_test = recast_at_center(fesolver,rbspace,rb_snaps_test,params_test)
post_process(info,feop,fesolver,snaps_test,params_test,approx_snaps_test,stats)

sol1 = approx_snaps_test[1]
sol1_ok = hcat(snaps_test[1:10]...)
