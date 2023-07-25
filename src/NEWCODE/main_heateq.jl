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

  h(x,μ,t) = abs(cos(t/μ[3]))
  h(μ,t) = x->h(x,μ,t)

  g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
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
  nsnaps_state = 80
  nsnaps_system = 30
  st_mdeim = false
  info = RBInfo(test_path;ϵ,load_offline,save_offline,energy_norm,nsnaps_state,nsnaps_system,st_mdeim)
  rbsolver = Backslash()
end

rbop = if load_offline
  load(GenericRBOperator,info)
else
  reduce_fe_operator(info,feop,fesolver)
end
test_rb_operator(info,feop,rbop,fesolver,rbsolver)

@everywhere begin
  root = pwd()
  include("$root/src/NEWCODE/FEM/FEM.jl")
  include("$root/src/NEWCODE/ROM/ROM.jl")
  include("$root/src/NEWCODE/RBTests.jl")
end

ϵ = info.ϵ
nsnaps = info.nsnaps_state
params = realization(feop,nsnaps)
sols = collect_solutions(feop,fesolver,params)
rbspace = compress_solutions(feop,fesolver,sols,params;ϵ)
trians = _collect_trian_res(feop)
r1 = collect_residuals(feop,fesolver,sols[1],params[1:1],trians[1],(1,1))
r2 = collect_residuals(feop,fesolver,sols[1],params[1:1],trians[2],(1,1))

r = collect_residuals(feop,fesolver,sres,pres,trian,(1,1))
begin
  op,solver = feop,fesolver
  cache = allocate_cache(op)
  times = get_times(solver)
  tdofs = length(times)
  r = allocate_residual(op,cache)
  trians = _collect_trian_res(feop)
  trian = trians[1]
  res_iter = init_vec_iterator(op,solver,trian,(1,1))
  times = get_times(solver)
  tdofs = length(times)
  # ye = map(enumerate(params)) do (nμ,μ)
  #   qt = map(times) do t
  #     update!(res_iter,op,solver,μ,t)
  #     evaluate!(res_iter)
  #   end
  #   hcat(qt...)
  # end
  row,_ = 1,1
  test_row = get_test(op)[row]
  dv_row = _get_fe_basis(op.test,row)
  assem_row = SparseMatrixAssembler(test_row,test_row)
  r = allocate_residual(op;assem=assem_row)
  # μ,t = params[1],dt
  update_cache!(cache,op,μ,t)
  u = evaluation_function(op,(zeros(647),zeros(647)),cache)
  vecdata = collect_cell_vector(test_row,op.res(μ,t,u,dv_row))#,trian)
  rnew = zeros(647)
  assemble_vector_add!(rnew,assem_row,vecdata)
end

cache = allocate_cache(feop)
update_cache!(cache,feop,μ,t)
boh
# # test_rb_operator(info,feop,rbop,fesolver,rbsolver)
# nsnaps_test=10
# sols,params = load_test(info,feop,fesolver)
# rb_res = Vector{RBResults}(undef,nsnaps_test)
# μ = params[1]
# urb,wall_time = solve(rbsolver,rbop,μ)
# rbres = RBResults(get_datum(sols[1]),urb,wall_time)
# for (u,μ) in zip(sols_test,params_test)
#   ic = initial_condition(sols,params,μ)
#   urb,wall_time = solve(rbsolver,rbop,μ)
#   push!(res,RBResults(u,urb,wall_time;kwargs...))
# end

# trians = _collect_trian_jac(feop)
# cjac = RBAlgebraicContribution()
# for trian in trians
#   ad_jac = compress_jacobians(feop,solver,trian,s,params,rbspace;
#     nsnaps=info.nsnaps_mdeim)
#   add_contribution!(cjac,trian,ad_jac)
# end
# cres = RBAlgebraicContribution()
# for trian in trians
#   ad_res = compress_residuals(feop,solver,trian,s,params,rbspace;
#     nsnaps=info.nsnaps_mdeim)
#   add_contribution!(cres,trian,ad_res)
# end

# times = get_times(solver)
# sols = get_data(s)
# trians = _collect_trian_jac(feop)
# matdatum = _matdata_jacobian(feop,solver,sols,param,trians[1])
# jacs = generate_jacobians(feop,solver,trians[1],s,param,(1,1))
# compress_component(jacs,solver,trians[1],rbspace,rbspace;st_mdeim=true)

# bs,bt = tpod(jacs)
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
trian = get_triangulation(test)
res_ad = rb_res_c[trian][1]
measures = get_measures(rb_res_c)
mdeim_interp = res_ad.mdeim_interpolation
red_integr_res = assemble_residual(feop,fesolver,res_ad,measures,(1,1),μ)
x = similar(red_integr_res)
copyto!(x,mdeim_interp.factors\red_integr_res)
