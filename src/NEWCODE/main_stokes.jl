# driver for unsteady poisson
using MPI,MPIClusterManagers,Distributed
manager = MPIWorkerManager()
addprocs(4)

@everywhere using Pkg; Pkg.activate(".")

@everywhere begin
  root = pwd()
  include("$root/src/NEWCODE/FEM/FEM.jl")
  include("$root/src/NEWCODE/ROM/ROM.jl")
  include("$root/src/NEWCODE/RBTests.jl")

  mesh = "model_circle_2D_coarse.json"
  test_path = "$root/tests/stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
  order = 2
  degree = 4

  fepath = fem_path(test_path)
  mshpath = mesh_path(test_path,mesh)
  model = get_discrete_model(mshpath,bnd_info)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=["neumann"])
  dΓn = Measure(Γn,degree)

  ranges = fill([1.,10.],3)
  sampling = UniformSampling()
  pspace = ParamSpace(ranges,sampling)

  a(x,μ,t) = exp((sin(t)+cos(t))*x[1]/sum(μ))
  a(μ,t) = x->a(x,μ,t)

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)
  g0(x,μ,t) = VectorValue(0,0)
  g0(μ,t) = x->g0(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)

  res(μ,t,(u,p),(v,q),dΩ,dΓn) = (∫(v⋅∂t(u))dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ
    - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ)
  jac(μ,t,(u,p),(du,dp),(v,q),dΩ) = ∫(a(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
  jac_t(μ,t,(u,p),(dut,dpt),(v,q),dΩ) = ∫(v⋅dut)dΩ

  res(μ,t,u,v) = res(μ,t,u,v,dΩ,dΓn)
  jac(μ,t,u,du,v) = jac(μ,t,u,du,v,dΩ)
  jac_t(μ,t,u,dut,v) = jac_t(μ,t,u,dut,v,dΩ)

  reffe_u = Gridap.ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = Gridap.ReferenceFE(lagrangian,Float,order-1)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  trial_u = ParamTransientTrialFESpace(test_u,[g0,g])
  test_p = TestFESpace(model,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = ParamTransientMultiFieldFESpace([test_u,test_p])
  trial = ParamTransientMultiFieldFESpace([trial_u,trial_p])
  feop = ParamTransientAffineFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tF,dt,θ = 0.,0.3,0.005,1
  uh0(μ) = interpolate_everywhere(u0(μ),trial_u(μ,t0))
  ph0(μ) = interpolate_everywhere(p0(μ),trial_p(μ,t0))
  xh0(μ) = interpolate_everywhere([uh0(μ),ph0(μ)],trial(μ,t0))
  fesolver = θMethod(LUSolver(),t0,tF,dt,θ,xh0)

  ϵ = 1e-4
  compute_supremizers = true
  nsnaps = 2
  nsnaps_mdeim = 2
  save_offline = false
  load_offline = false
  energy_norm = false
  info = RBInfo(test_path;ϵ,load_offline,save_offline,energy_norm,nsnaps,nsnaps_mdeim)
end

ϵ = info.ϵ
nsnaps = info.nsnaps
params = realization(feop,nsnaps)
cache = solution_cache(feop.test,fesolver)
sols = generate_solutions(feop,fesolver,params)
rbspace = compress_solutions(feop,fesolver,sols,params;ϵ,compute_supremizers)
nsnaps = info.nsnaps_mdeim
rb_res = compress_residuals(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
rb_jac = compress_jacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)

# rbspace = reduce_fe_space(info,feop,fesolver;compute_supremizers=true)
# rbop = reduce_fe_operator(info,feop,fesolver,rbspace)
# rbsolver = Backslash()

# u_rb = solve(rbsolver,rbop;n_solutions=10,post_process=true,energy_norm)
