begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")

  mesh = "elasticity_3cyl2D.json"
  test_path = "$root/tests/poisson/unsteady/$mesh"
  bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
  order = 1
  degree = 2

  model = get_discrete_model(test_path,mesh,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ranges = fill([1.,2.],3)
  sampling = UniformSampling()
  pspace = ParamSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)

  f(x,μ,t) = 1.
  f(μ,t) = x->f(x,μ,t)

  h(x,μ,t) = abs(cos(t/μ[3]))
  h(μ,t) = x->h(x,μ,t)

  g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = 0
  u0(μ) = x->u0(x,μ)

  res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂ₚt(u))dΩ + ∫(a(μ,t)*∇(v)⋅∇(u))dΩ - ∫(f(μ,t)*v)dΩ - ∫(h(μ,t)*v)dΓn
  jac(μ,t,u,du,v,dΩ) = ∫(a(μ,t)*∇(v)⋅∇(du))dΩ
  jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

  res(μ,t,u,v) = res(μ,t,u,v,dΩ,dΓn)
  jac(μ,t,u,du,v) = jac(μ,t,u,du,v,dΩ)
  jac_t(μ,t,u,dut,v) = jac_t(μ,t,u,dut,v,dΩ)

  reffe = ReferenceFE(lagrangian,Float,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = ParamTransientTrialFESpace(test,g)
  feop = ParamTransientAffineFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tF,dt,θ = 0.,0.05,0.005,1
  uh0(μ) = interpolate_everywhere(u0(μ),trial(μ,t0))
  fesolver = θMethod(LUSolver(),t0,tF,dt,θ,uh0)

  ϵ = 1e-4
  save_structures = true
  load_structures = true
  energy_norm = false
  nsnaps_state = 80
  nsnaps_system = 30
  st_mdeim = false
  info = RBInfo(test_path;ϵ,load_structures,save_structures,energy_norm,
                nsnaps_state,nsnaps_system,st_mdeim)
  rbsolver = Backslash()
end

#OK
sols = collect_solutions(info,feop,fesolver)
#TRY
collector = CollectResidualsMap(fesolver,feop,trian)
nres = info.nsnaps_system
sols = get_snaps(snaps)
cache = array_cache(sols)

function run_collection(collector::CollectResidualsMap)
  params = get_params(snaps,1:nres)
  printstyled("Generating $nres residuals snapshots\n";color=:blue)
  ress = lazy_map(eachindex(params)) do i
    sol_i = getindex!(cache,sols,i)
    param_i = getindex(params,i)
    collector.f(sol_i,param_i)
  end
  return ress,params
end

function run_collection(collector::CollectResidualsMap{Union{TimeAffinity,NonAffinity}})
  params = get_params(snaps,1)
  printstyled("Generating 1 residual snapshot\n";color=:blue)
  ress = lazy_map(eachindex(params)) do i
    sol_i = getindex!(cache,sols,i)
    param_i = getindex(params,i)
    collector.f(sol_i,param_i)
  end
  return ress,params
end

collector = CollectResidualsMap(fesolver,feop,trian)
ress,params = lazy_map(collector,sols)
Snapshots(collector,ress,params)
