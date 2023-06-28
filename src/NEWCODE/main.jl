# driver for unsteady poisson
using MPI,MPIClusterManagers,Distributed
manager = MPIWorkerManager()
addprocs(manager)

@everywhere begin
  root = pwd()
  include("$root/src/NEWCODE/FEM/FEM.jl")
  include("$root/src/NEWCODE/ROM/ROM.jl")
  include("$root/src/NEWCODE/RBTests.jl")

  mesh = "elasticity_3cyl2D.json"
  test_path = "$root/tests/poisson/unsteady/$mesh"
  bnd_info = Dict("dirichlet" => ["dirichlet"],"neumann" => ["neumann"])
  order = 1
  degree = 2

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

  f(x,μ,t) = 1.
  f(μ,t) = x->f(x,μ,t)

  h(x,μ,t) = abs(cos(μ[3]*t))
  h(μ,t) = x->h(x,μ,t)

  g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(μ[3]*t))
  g(μ,t) = x->g(x,μ,t)

  u0(x,μ) = 0
  u0(μ) = x->u0(x,μ)

  res(μ,t,u,v,dΩ,dΓn) = ∫(v*∂t(u))dΩ + ∫(a(μ,t)*∇(v)⋅∇(u))dΩ - ∫(f(μ,t)*v)dΩ - ∫(h(μ,t)*v)dΓn
  jac(μ,t,u,du,v,dΩ) = ∫(a(μ,t)*∇(v)⋅∇(du))dΩ
  jac_t(μ,t,u,dut,v,dΩ) = ∫(v*dut)dΩ

  res(μ,t,u,v) = res(μ,t,u,v,dΩ,dΓn)
  jac(μ,t,u,du,v) = jac(μ,t,u,du,v,dΩ)
  jac_t(μ,t,u,dut,v) = jac_t(μ,t,u,dut,v,dΩ)

  reffe = Gridap.ReferenceFE(lagrangian,Float,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
  trial = ParamTransientTrialFESpace(test,g)
  feop = ParamTransientAffineFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tF,dt,θ = 0.,0.05,0.005,1
  uh0(μ) = interpolate_everywhere(u0(μ),trial(μ,t0))
  fesolver = θMethod(LUSolver(),t0,tF,dt,θ,uh0)

  ϵ = 1e-4
  save_offline = true
  load_offline = true
  energy_norm = false
  nsnaps = 10
  nsnaps_mdeim = 10
  info = RBInfo(test_path;ϵ,load_offline,save_offline,energy_norm,nsnaps,nsnaps_mdeim)
end

ϵ = info.ϵ
nsnaps = info.nsnaps
params = realization(feop,nsnaps)
sols = generate_solutions(feop,fesolver,params)
rbspace = compress_solution(sols,feop,fesolver;ϵ)

nsnaps = info.nsnaps_mdeim
trians = _collect_trian_res(feop)
trian = first(trians)
vecdata = _vecdata_residual(feop,fesolver,trian,get_data(sols),params,(1,1))
aff = get_affinity(fesolver,params,vecdata)
data = get_data(aff,fesolver,params,vecdata)
cache = residuals_cache(feop.assem,data)
ress = pmap(d -> collect_residuals!(cache,feop,d),data)
dummy
# solver = fesolver
# sols,params = solve(fesolver,feop,nsnaps)
# cache = solution_cache(feop.test,fesolver)
# snaps = pmap(sol->collect_solution!(cache,sol),sols)
# s = Snapshots(snaps)
# param = Table(params)

# trians = _collect_trian_jac(feop)
# cjac = RBAlgebraicContribution()
# for trian in trians
#   ad_jac = compress_jacobian(feop,solver,trian,s,params,rbspace;
#     nsnaps=info.nsnaps_mdeim)
#   add_contribution!(cjac,trian,ad_jac)
# end
# cres = RBAlgebraicContribution()
# for trian in trians
#   ad_res = compress_residual(feop,solver,trian,s,params,rbspace;
#     nsnaps=info.nsnaps_mdeim)
#   add_contribution!(cres,trian,ad_res)
# end

# times = get_times(solver)
# sols = get_data(s)
# trians = _collect_trian_jac(feop)
# matdatum = _matdata_jacobian(feop,solver,sols,param,trians[1])
# jacs = generate_jacobians(feop,solver,trians[1],s,param,(1,1))
# compress_component(jacs,solver,trians[1],rbspace,rbspace;st_mdeim=true)

# bs,bt = transient_tpod(jacs)
# interp_idx_space,interp_idx_time = get_interpolation_idx(bs,bt)
# integr_domain = TransientRBIntegrationDomain(
#   jacs,trians[1],times,interp_idx_space,interp_idx_time)
# interp_bs = bs[interp_idx_space,:]
# interp_bt = bt[interp_idx_time,:]
# interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
# lu_interp_bst = lu(interp_bst)
# proj_bs,proj_bt = compress(solver,bs,bt,rbspace,rbspace)
# TransientRBAffineDecomposition(proj_bs,proj_bt,lu_interp_bst,integr_domain)

# if load_offline
#   rbop = load(RBOperator,info)
# else
#   rbspace = reduce_fe_space(info,feop,fesolver)
#   rbop = reduce_fe_operator(info,feop,fesolver,rbspace)
# end

# rbsolver = Backslash()
# u_rb = solve(info,rbsolver,rbop;n_solutions=10,post_process=true,energy_norm)
