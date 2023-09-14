begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")

  mesh = "model_circle_2D_coarse.json"
  test_path = "$root/tests/stokes/unsteady/$mesh"
  bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
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

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)
  g0(x,μ,t) = VectorValue(0,0)
  g0(μ,t) = x->g0(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)

  res(μ,t,(u,p),(v,q),dΩ) = (∫(v⋅∂ₚt(u))dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ
    - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ)
  jac(μ,t,(u,p),(du,dp),(v,q),dΩ) = ∫(a(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
  jac_t(μ,t,(u,p),(dut,dpt),(v,q),dΩ) = ∫(v⋅dut)dΩ

  res(μ,t,u,v) = res(μ,t,u,v,dΩ)
  jac(μ,t,u,du,v) = jac(μ,t,u,du,v,dΩ)
  jac_t(μ,t,u,dut,v) = jac_t(μ,t,u,dut,v,dΩ)

  reffe_u = Gridap.ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = Gridap.ReferenceFE(lagrangian,Float,order-1)
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  trial_u = PTTrialFESpace(test_u,[g0,g])
  test_p = TestFESpace(model,reffe_p;conformity=:C0)
  trial_p = TrialFESpace(test_p)
  test = PTMultiFieldFESpace([test_u,test_p])
  trial = PTMultiFieldFESpace([trial_u,trial_p])
  feop = PTAffineFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tF,dt,θ = 0.,0.05,0.005,1
  uh0(μ) = interpolate_everywhere(u0(μ),trial_u(μ,t0))
  ph0(μ) = interpolate_everywhere(p0(μ),trial_p(μ,t0))
  xh0(μ) = interpolate_everywhere([uh0(μ),ph0(μ)],trial(μ,t0))
  fesolver = ThetaMethod(LUSolver(),t0,tF,dt,θ,xh0)

  ϵ = 1e-4
  compute_supremizers = true
  nsnaps_state = 50
  nsnaps_system = 20
  save_structures = true
  load_structures = true
  energy_norm = false
  st_mdeim = true
  info = RBInfo(test_path;ϵ,save_structures,load_structures,energy_norm,
                nsnaps_state,nsnaps_system,st_mdeim)
end

nsols = info.nsnaps_state
params = realization(feop,nsols)
sols = collect_solutions(info,feop,fesolver,params)

nsnaps = info.nsnaps_system
rb_res_c = compress_residuals(feop,fesolver,rbspace,sols,params;ϵ,nsnaps,st_mdeim)
rb_jac_c = compress_jacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps,st_mdeim)
rb_djac_c = compress_djacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps,st_mdeim)
rb_res = collect_residual_contributions(feop,fesolver,rb_res_c;st_mdeim)
rb_jac = collect_jacobian_contributions(feop,fesolver,rb_jac_c;st_mdeim)
rb_djac = collect_jacobian_contributions(feop,fesolver,rb_djac_c;i=2,st_mdeim)
