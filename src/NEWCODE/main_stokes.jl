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

  res(μ,t,(u,p),(v,q),dΩ) = (∫(v⋅∂t(u))dΩ + ∫(a(μ,t)*∇(v)⊙∇(u))dΩ
    - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ)
  jac(μ,t,(u,p),(du,dp),(v,q),dΩ) = ∫(a(μ,t)*∇(v)⊙∇(du))dΩ - ∫(dp*(∇⋅(v)))dΩ - ∫(q*(∇⋅(du)))dΩ
  jac_t(μ,t,(u,p),(dut,dpt),(v,q),dΩ) = ∫(v⋅dut)dΩ

  res(μ,t,u,v) = res(μ,t,u,v,dΩ)
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
  nsnaps_state = 50
  nsnaps_system = 20
  save_offline = true
  load_offline = false
  energy_norm = false
  st_mdeim = true
  info = RBInfo(test_path;ϵ,load_offline,save_offline,energy_norm,nsnaps_state,nsnaps_system)
end

ϵ = info.ϵ
nsnaps = info.nsnaps_state
params = realization(feop,nsnaps)
cache = solution_cache(feop.test,fesolver)
sols = generate_solutions(feop,fesolver,params)
rbspace = compress_solutions(feop,fesolver,sols,params;ϵ,compute_supremizers)
rb_res_c = compress_residuals(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
rb_jac_c = compress_jacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
rb_res = collect_residual_contributions(feop,fesolver,rb_res_c;st_mdeim)
rb_jac = collect_jacobian_contributions(feop,fesolver,rb_jac_c;st_mdeim)
test_rb_operator(info,feop,rbop,fesolver,rbsolver)

global r
for (m,ad) in rb_res_c.dict
  nfields = length(ad)
  trian = get_triangulation(m)
  r = Vector{ParamArray}(undef,nfields)
  for row = 1:nfields
    ad_row = ad[row]
    r[row] = residual_contribution(feop,fesolver,ad_row,(row,1),trian,
      (m,);st_mdeim)
  end
  r
end

global j
for (m,ad) in rb_jac_c.dict
  nfields = size(ad,1)
  trian = get_triangulation(m)
  j = Matrix{ParamArray}(undef,nfields,nfields)
  for row = 1:nfields, col=1:nfields
    ad_row = ad[row,col]
    j[row,col] = jacobian_contribution(feop,fesolver,ad_row,(row,col),trian,
      (m,);st_mdeim)
  end
  j
end

# m = get_domains(rb_res_c)
# trian = get_triangulation(m...)
# nfields = 2
# r = Vector{ParamArray}(undef,nfields)
# row = 1
# ad = rb_res_c[m...]
# ad_row = ad[row]
# coeff = residual_coefficient(feop,fesolver,ad_row,(row,1),trian,m;st_mdeim)
# u = get_datum(sols[1])
# μ = params[1]
# input = (u,μ)
# red_integr_res = assemble_residual(feop,fesolver,ad_row,input,(1,1),trian,m)
# coeff = solve(ad_row.mdeim_interpolation,reshape(red_integr_res,:))
# rcoeff = recast_coefficient(ad_row.basis_time,coeff)
# _,bt = ad_row.basis_time
# time_ndofs = size(bt,1)
# proj = map(eachcol(rcoeff)) do c
#   pc = map(eachcol(bt)) do b
#     sum(b.*c)
#   end
#   reshape(pc,:,1)
# end
# proj

m = get_domains(rb_jac_c)
trian = get_triangulation(m...)
nfields = 2
row,col = 2,1
ad = rb_jac_c[m...]
ad_row = ad[row,col]
coeff = jacobian_coefficient(feop,fesolver,ad_row,(row,col),trian,m;st_mdeim)
u = get_datum(sols[1])
μ = params[1]
input = (u,μ)
red_integr_res = assemble_jacobian(feop,fesolver,ad_row,input,(row,col),trian,m...)
coeff = solve(ad_row.mdeim_interpolation,reshape(red_integr_res,:))
rcoeff = recast_coefficient(ad_row.basis_time,coeff)
_,btbt,btbt_shift = ad_row.basis_time
projs = Matrix{Float}[]
@inbounds for q = axes(rcoeff,2), ijt = axes(btbt,2)
  proj = sum(btbt[:,ijt].*rcoeff[:,q])
  proj_shift = sum(btbt_shift[:,ijt].*coeff[2:end,q])
  push!(projs,proj+proj_shift)
end
projs

row,col = 2,1
trian = get_triangulation(feop.test[row])
sjac = get_datum(sols[1:nsnaps])
pjac = params[1:nsnaps]
cell_dof_ids = get_cell_dof_ids(feop.test[row],trian)
order = get_order(feop.test[row])

matdata = _matdata_jacobian(feop,fesolver,sjac,pjac,(row,col);trian)
aff = get_affinity(fesolver,pjac,matdata)
data = get_datum(aff,fesolver,pjac,matdata)
cache = jacobians_cache(feop.assem,data)
jacs = pmap(d -> collect_jacobians!(cache,feop.assem,d),data)
njacs = _get_nsnaps(aff,params)
j = generate_jacobians(feop,fesolver,pjac,matdata)
bs,bt = transient_tpod(j,fesolver;ϵ)
interp_idx_space,interp_idx_time = get_interpolation_idx(bs,bt)

# nsnaps = info.nsnaps_mdeim
# rb_res = compress_residuals(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)
# rb_jac = compress_jacobians(feop,fesolver,rbspace,sols,params;ϵ,nsnaps)

# rbspace = reduce_fe_space(info,feop,fesolver;compute_supremizers=true)
# rbop = reduce_fe_operator(info,feop,fesolver,rbspace)
# rbsolver = Backslash()

# u_rb = solve(rbsolver,rbop;n_solutions=10,post_process=true,energy_norm)
