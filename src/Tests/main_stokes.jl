begin
  root = pwd()
  include("$root/src/Utils/Utils.jl")
  include("$root/src/FEM/FEM.jl")
  include("$root/src/RB/RB.jl")
end

begin
  # mesh = "model_circle_2D_coarse.json"
  mesh = "cube2x2.json"
  test_path = "$root/tests/stokes/unsteady/$mesh"
  # bnd_info = Dict("dirichlet0" => ["noslip"],"dirichlet" => ["inlet"],"neumann" => ["outlet"])
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

  f(x,μ,t) = VectorValue(0,0)
  f(μ,t) = x->f(x,μ,t)
  fμt(μ,t) = PTFunction(f,μ,t)

  g(x,μ,t) = VectorValue(μ[1]*exp(-x[2]/μ[2])*abs(sin(μ[3]*t)),0)
  g(μ,t) = x->g(x,μ,t)
  # g0(x,μ,t) = VectorValue(0,0)
  # g0(μ,t) = x->g0(x,μ,t)

  u0(x,μ) = VectorValue(0,0)
  u0(μ) = x->u0(x,μ)
  u0μ(μ) = PFunction(u0,μ)
  p0(x,μ) = 0
  p0(μ) = x->p0(x,μ)
  p0μ(μ) = PFunction(p0,μ)

  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = ∫ₚ(v⋅dut,dΩ)
  jac(μ,t,(u,p),(du,dp),(v,q)) = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ) - ∫ₚ(dp*(∇⋅(v)),dΩ) - ∫ₚ(q*(∇⋅(du)),dΩ)
  res(μ,t,(u,p),(v,q)) = (∫ₚ(v⋅∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(p*(∇⋅(v)),dΩ)
    - ∫ₚ(q*(∇⋅(u)),dΩ) - ∫ₚ(v⋅fμt(μ,t),dΩ))

  reffe_u = Gridap.ReferenceFE(lagrangian,VectorValue{2,Float},order)
  reffe_p = Gridap.ReferenceFE(lagrangian,Float,order-1)
  # test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet0","dirichlet"])
  test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet"])
  # trial_u = PTTrialFESpace(test_u,[g0,g])
  trial_u = PTTrialFESpace(test_u,g)
  test_p = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
  trial_p = TrialFESpace(test_p)
  test = PTMultiFieldFESpace([test_u,test_p])
  trial = PTMultiFieldFESpace([trial_u,trial_p])
  feop = PTAffineFEOperator(res,jac,jac_t,pspace,trial,test)
  t0,tf,dt,θ = 0.,0.05,0.005,1
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial_u(μ,t0))
  ph0μ(μ) = interpolate_everywhere(p0μ(μ),trial_p(μ,t0))
  xh0μ(μ) = interpolate_everywhere([uh0μ(μ),ph0μ(μ)],trial(μ,t0))
  fesolver = PThetaMethod(LUSolver(),xh0μ,θ,dt,t0,tf)

  ϵ = 1e-4
  load_solutions = true
  save_solutions = true
  load_structures = false
  save_structures = true
  energy_norm = [:l2,:l2]
  compute_supremizers = true
  nsnaps_state = 50
  nsnaps_system = 20
  nsnaps_test = 10
  st_mdeim = true
  info = RBInfo(test_path;ϵ,load_solutions,save_solutions,load_structures,save_structures,
                energy_norm,compute_supremizers,nsnaps_state,nsnaps_system,nsnaps_test,st_mdeim)
  # multi_field_rb_model(info,feop,fesolver)
end

sols,params = load(info,(BlockSnapshots,Table))
rbspace = load(info,BlockRBSpace)
rbrhs,rblhs = load(info,(BlockRBVecAlgebraicContribution,Vector{BlockRBMatAlgebraicContribution}))

nsnaps = info.nsnaps_state
params = realization(feop,nsnaps)
sols,stats = collect_solutions(fesolver,feop,trial,params)
save(info,(sols,params,stats))
rbspace = reduced_basis(info,feop,sols,params)
rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
save(info,(rbspace,rbrhs,rblhs))

snaps_test,params_test = load_test(info,feop,fesolver)
x = initial_guess(sols,params,params_test)
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,snaps_test,params_test)
rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,x,params_test)
lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,x,params_test)
stats = @timed begin
  rb_snaps_test = rb_solve(fesolver.nls,rhs,lhs)
end
approx_snaps_test = recast(rbspace,rb_snaps_test)

# lhs = collect_lhs_contributions!(lhs_cache,info,feop,fesolver,rblhs,rbspace,x,params_test)
njacs = length(rblhs)
nblocks = get_nblocks(testitem(rblhs))
rb_jacs_contribs = Vector{PTArray{Matrix{Float}}}(undef,njacs)
i = 1
row,col = 1,1

rb_jac_i = rblhs[i]
feop_row_col = feop[row,col]
sols_col = x[col]
rb_jac_i_row_col,touched_i_row_col = rb_jac_i[row,col]
rbspace_row = rbspace[row]
rbspace_col = rbspace[col]
# collect_lhs_contributions!(lhs_cache,info,feop_row_col,fesolver,rb_jac_i_row_col,rbspace_col,sols_col,params_test;i)
coeff_cache,rb_cache = lhs_cache
trian = [get_domains(rb_jac_i_row_col)...]
st_mdeim = info.st_mdeim
times = get_times(fesolver)

rb_jac_contribs = Vector{PTArray{Matrix{Float}}}(undef,num_domains(rb_jac_i_row_col))
rbjact = rb_jac_i_row_col[Ω]
# coeff = lhs_coefficient!(coeff_cache,feop_row_col,fesolver,rbjact,sols_col,params_test;st_mdeim,i)
  jcache = coeff_cache[1]
  ndofs_row = num_free_dofs(feop_row_col.test)
  ndofs_col = num_free_dofs(get_trial(feop_row_col)(nothing,nothing))
  setsize!(jcache,(ndofs_row,ndofs_col))

  red_idx = rbjact.integration_domain.idx
  red_times = rbjact.integration_domain.times
  red_meas = rbjact.integration_domain.meas

  A = get_array(jcache;len=length(red_times)*length(params_test))
  _sols = get_solutions_at_times(sols_col,fesolver,times)

  # collect_jacobians_for_idx!(A,fesolver,feop_row_col,_sols,params_test,times,red_idx,red_meas;i)
  dt,θ = fesolver.dt,fesolver.θ
  dtθ = θ == 0.0 ? dt : dt*θ
  ode_op = get_algebraic_operator(feop_row_col)
  ode_cache = allocate_cache(ode_op,params_test,times)
  ode_cache = update_cache!(ode_cache,ode_op,params_test,times)
  sols_cache = copy(_sols) .* 0.
  nlop = get_nonlinear_operator(ode_op,params_test,times,dtθ,_sols,ode_cache,sols_cache)
  uF = _sols
  vθ = nlop.vθ
  @. vθ = (_sols-nlop.u0)/nlop.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(feop_row_col)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],(vθ,vθ)[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],(vθ,vθ)[1]),dxh)
  Uh = get_trial(feop_row_col)(params_test,times)
  V = get_test(feop_row_col)
  u = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  dc = feop_row_col.jacs[i](params_test,times,xh,u,v)[dΩ]
  # matdata = collect_cell_matrix(Uh,V,dc)
    w = []
    r = []
    c = []
    strian = [get_domains(dc)...][1]
    scell_mat = get_contribution(dc,strian)
    cell_mat, trian = move_contributions(scell_mat,strian)
    @assert ndims(eltype(cell_mat)) == 2
    cell_mat_c = attach_constraints_cols(Uh,cell_mat,trian)
    cell_mat_rc = attach_constraints_rows(V,cell_mat_c,trian)
    rows = get_cell_dof_ids(V,trian)
    cols = get_cell_dof_ids(Uh,trian)
    push!(w,cell_mat_rc)
    push!(r,rows)
    push!(c,cols)
  assemble_matrix_add!(A,feop_row_col.assem,(w,r,c))


__jacs,__trian = collect_jacobians_for_trian(fesolver,feop_row_col,_sols,params_test,times;i)
  _ode_op = get_algebraic_operator(feop_row_col)
  _ode_cache = allocate_cache(ode_op,params_test,times)
  _A = allocate_jacobian(ode_op,params_test,times,_sols,ode_cache)
  _ode_cache = update_cache!(ode_cache,ode_op,params_test,times)
  _sols_cache = copy(_sols) .* 0.
  _nlop = get_nonlinear_operator(_ode_op,params_test,times,dtθ,_sols,_ode_cache,_sols_cache)
  _vθ = _nlop.vθ
  _z = zero(eltype(_A))
  fillstored!(_A,_z)
  _Xh, = _ode_cache
  _dxh = ()
  for i in 2:get_order(feop_row_col)+1
    _dxh = (_dxh...,EvaluationFunction(_Xh[i],(_sols,_sols)[i]))
  end
  _xh=TransientCellField(EvaluationFunction(Xh[1],(_sols,_sols)[1]),dxh)
  _Uh = get_trial(feop_row_col)(params_test,times)
  _V = get_test(feop_row_col)
  _u = get_trial_fe_basis(_Uh)
  _v = get_fe_basis(_V)
  _dc = integrate(feop_row_col.jacs[i](params_test,times,xh,u,v))
  _trian = get_domains(_dc)
  t = [_trian...][1]
  _matdata = collect_cell_matrix(_Uh,_V,_dc,t)
  assemble_matrix_add!(_A,feop_row_col.assem,_matdata)

full_ode_op = get_algebraic_operator(feop)
full_ode_cache = allocate_cache(full_ode_op,params_test,times)
M = allocate_jacobian(full_ode_op,params_test,times,vcat(x...),full_ode_cache)[1]
Nc = CachedArray(M)
Mboh = setsize!(Mc,(24,24))
