# LOW LEVEL
nsnaps = info.nsnaps_system
snapsθ = recenter(fesolver,sols,params)
for i = eachindex(snapsθ)
  test_ptarray(snapsθ.snaps[i],sols.snaps[i])
end
_snapsθ,_μ = snapsθ[1:nsnaps],params[1:nsnaps]
times = get_times(fesolver)
ress,trian = collect_residuals_for_trian(fesolver,feop,_snapsθ,_μ,times)
  # ode_op = get_algebraic_operator(feop)
  # ode_cache = allocate_cache(ode_op,params,times)
  # b = allocate_residual(ode_op,_snapsθ,ode_cache)
  # dt,θ = fesolver.dt,fesolver.θ
  # dtθ = θ == 0.0 ? dt : dt*θ
  # ode_cache = update_cache!(ode_cache,ode_op,params,times)
  # nlop = PThetaMethodNonlinearOperator(ode_op,params,times,dtθ,_snapsθ,ode_cache,_snapsθ)
  # xhF = (_snapsθ,_snapsθ-_snapsθ)
  # Xh, = ode_cache
  # dxh = ()
  # for i in 2:get_order(ode_op)+1
  #   dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  # end
  # xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  # V = get_test(feop)
  # v = get_fe_basis(V)
  # dc = feop.res(params,times,xh,v)
  # trian = get_domains(dc)
  # bvec = Vector{typeof(b)}(undef,num_domains(dc))
  # for (n,t) in enumerate(trian)
  #   vecdata = collect_cell_vector(V,dc,t)
  #   assemble_vector_add!(b,feop.assem,vecdata)
  #   bvec[n] = copy(b)
  # end
  # nzm = NnzMatrix(bvec[1];nparams=length(params))
nzm,_trian = ress[2],Γn
full_val = recast(nzm)
basis_space,basis_time = compress(nzm;ϵ=info.ϵ)
proj_bs,proj_bt = compress_space_time(basis_space,basis_time,rbspace)
interp_idx_space,interp_idx_time = get_interpolation_idx(basis_space),get_interpolation_idx(basis_time)
entire_interp_idx_space = recast_idx(basis_space,interp_idx_space)

maximum(abs.(full_val - basis_space*basis_space'*full_val)) <= ϵ*10
m2 = change_mode(nzm)
maximum(abs.(m2 - basis_time*basis_time'*m2)) <= ϵ*10

interp_bs = basis_space[interp_idx_space,:]
lu_interp = if info.st_mdeim
  interp_bt = basis_time[interp_idx_time,:]
  interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
  lu(interp_bst)
else
  lu(interp_bs)
end

cell_dof_ids = get_cell_dof_ids(feop.test,_trian)
red_integr_cells = find_cells(entire_interp_idx_space,cell_dof_ids)
red_trian = view(_trian,red_integr_cells)
red_times = st_mdeim ? times[interp_idx_time] : times

i = 1
# ode_op = get_algebraic_operator(feop)
# ode_cache = allocate_cache(ode_op,params,times)
# A = allocate_jacobian(ode_op,_snapsθ,ode_cache)
# dt,θ = fesolver.dt,fesolver.θ
# dtθ = θ == 0.0 ? dt : dt*θ
# ode_cache = update_cache!(ode_cache,ode_op,params,times)
# nlop = PThetaMethodNonlinearOperator(ode_op,params,times,dtθ,_snapsθ,ode_cache,_snapsθ)
# @which collect_jacobians!(Val(true),A,nlop,_snapsθ;i)
# jacs_i,trian = jacobian!(A,nlop,_snapsθ,i,Val(true))
# @which NnzMatrix(map(NnzVector,jacs_i[1]);nparams=length(nlop.μ))
# nzm = NnzMatrix.(jacs_i;nparams=length(nlop.μ))
times = get_times(fesolver)
combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : x-y
jacs,trian = collect_jacobians_for_trian(fesolver,feop,_snapsθ,_μ,times;i)
# compress_component(info,feop,jacs,trian,times,rbspace,rbspace;combine_projections)
  basis_space,basis_time = compress(jacs[1];ϵ=info.ϵ)
  proj_bs,proj_bt = compress_space_time(basis_space,basis_time,rbspace,rbspace;combine_projections)

  interp_idx_space,interp_idx_time = get_interpolation_idx(basis_space),get_interpolation_idx(basis_time)
  entire_interp_idx_space = recast_idx(basis_space,interp_idx_space)

  maximum(abs.(full_val - basis_space*basis_space'*full_val)) <= ϵ*10
  m2 = change_mode(nzm)
  maximum(abs.(m2 - basis_time*basis_time'*m2)) <= ϵ*10

  interp_bs = basis_space[interp_idx_space,:]
  lu_interp = if info.st_mdeim
    interp_bt = basis_time[interp_idx_time,:]
    interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
    lu(interp_bst)
  else
    lu(interp_bs)
  end

function test_affine_decomposition_rhs(
  cache,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbres::RBAffineDecomposition,
  meas::Measure,
  sols::PTArray,
  μ::Table,
  offline_sols::Snapshots,
  offline_params::Table)

  rcache,scache... = cache

  times = get_times(fesolver)
  ndofs = num_free_dofs(feop.test)
  setsize!(rcache,(ndofs,))

  red_idx = rbres.integration_domain.idx
  red_times = rbres.integration_domain.times
  red_meas = rbres.integration_domain.meas
  full_idx = collect(1:test.nfree)

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,red_times)

  if length(red_times) < length(times)
    b = get_array(rcache;len=length(red_times)*length(μ))
    time_idx = findall(x->x in red_times,times)
    idx = param_time_idx(time_idx,length(μ))
    _sols = PTArray(sols[idx])
  else
    b = get_array(rcache)
    _sols = sols
  end
  bfull = copy(b)
  res = collect_residuals!(b,fesolver,ode_op,_sols,μ,red_times,ode_cache,red_idx,red_meas)
  res_full = collect_residuals!(bfull,fesolver,ode_op,_sols,μ,red_times,ode_cache,full_idx,meas)
  res_offline,_ = collect_residuals_for_trian(fesolver,feop,offline_sols[1:30],offline_params[1:30],times)

  err_res = maximum(abs.(res-res_full[red_idx,:]))
  println("Residual difference for selected triangulation is $err_res")

  coeff = mdeim_solve!(scache[1],rbres.mdeim_interpolation,res)
  try
    basis_space = tpod(res_offline[1])
    coeff_ok = basis_space'*res_full
  catch
    basis_space = tpod(res_offline[2])
    coeff_ok = basis_space'*res_full
  end
  err_coeff = maximum(abs.(coeff-coeff_ok))
  println("Residual coefficient difference for selected triangulation is $err_coeff")
end

rbres,rbjac = rbrhs,rblhs
sols_test,params_test = load_test(info,feop,fesolver)
res_cache,jac_cache = allocate_sys_cache(feop,fesolver,rbspace,sols_test,params_test)

rbrest = rbres[Ω]
meas = dΩ
cache = res_cache[1]

test_affine_decomposition_rhs(cache,feop,fesolver,rbrest,meas,sols_test,params_test,sols,params)

# rcache,scache... = cache

# times = get_times(fesolver)
# ndofs = num_free_dofs(feop.test)
# setsize!(rcache,(ndofs,))

# red_idx = rbrest.integration_domain.idx
# red_times = rbrest.integration_domain.times
# red_meas = rbrest.integration_domain.meas
# full_idx = collect(1:test.nfree)

# ode_op = get_algebraic_operator(feop)
# ode_cache = allocate_cache(ode_op,params_test,red_times)
# b = get_array(rcache)
# bfull = copy(b)
# Res = collect_residuals!(b,fesolver,ode_op,sols_test,params_test,red_times,ode_cache,red_idx,red_meas)
# Res_full = collect_residuals!(bfull,fesolver,ode_op,sols_test,params_test,red_times,ode_cache,full_idx,meas)

# err_res = maximum(abs.(Res-Res_full[red_idx,:]))
# println("Resiudal difference for selected triangulation is $err_res")

# #coeff = mdeim_solve!(scache,rbres,res;st_mdeim)
# coeff = mdeim_solve!(scache[1],rbrest.mdeim_interpolation,Res)
# basis_space = tpod(Res_full)
# coeff_ok = basis_space'*Res_full

# function _alt_mdeim_solve(lu,A)
#   P_A = lu.P*A
#   y = lu.L \ P_A
#   x = lu.U \ y
#   x
# end

# _coeff = _alt_mdeim_solve(rbrest.mdeim_interpolation,Res)
# @check coeff ≈ _coeff
# _coeff_ok = basis_space[red_idx]'*Res_full


function test_affine_decomposition_lhs(
  cache,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbjac::RBAffineDecomposition,
  meas::Measure,
  sols::PTArray,
  μ::Table,
  offline_sols::Snapshots,
  offline_params::Table;
  i=1)

  jcache,scache... = cache

  times = get_times(fesolver)
  ndofs_row = num_free_dofs(feop.test)
  ndofs_col = num_free_dofs(get_trial(feop)(nothing,nothing))
  setsize!(jcache,(ndofs_row,ndofs_col))

  red_idx = rbjac.integration_domain.idx
  red_times = rbjac.integration_domain.times
  red_meas = rbjac.integration_domain.meas

  #test_red_meas(feop,fesolver,sols,μ,red_meas)

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,red_times)

  if length(red_times) < length(times)
    A = get_array(jcache;len=length(red_times)*length(μ))
    time_idx = findall(x->x in red_times,times)
    idx = param_time_idx(time_idx,length(μ))
    _sols = PTArray(sols[idx])
  else
    A = get_array(jcache)
    _sols = sols
  end
  Afull = copy(A)
  full_idx = findnz(Afull[1][:])[1]
  jac = collect_jacobians!(A,fesolver,ode_op,_sols,μ,red_times,ode_cache,red_idx,red_meas;i)
  jac_full = collect_jacobians!(Afull,fesolver,ode_op,_sols,μ,red_times,ode_cache,full_idx,meas;i)
  jac_offline,_ = collect_jacobians_for_trian(fesolver,feop,offline_sols[1:30],offline_params[1:30],times;i)
  basis_space = tpod(jac_offline[1])
  interp_idx_space = get_interpolation_idx(basis_space)
  err_jac = maximum(abs.(Jac-Jac_full[interp_idx_space,:]))
  println("Jacobian #$i difference for selected triangulation is $err_jac")

  coeff = mdeim_solve!(scache[1],rbjac.mdeim_interpolation,jac)
  coeff_ok = basis_space'*jac_full
  err_coeff = maximum(abs.(coeff-coeff_ok))
  println("Jacobian #$i coefficient difference for selected triangulation is $err_coeff")
end

i = 1
rbjact = rbjac[i][Ω]

meas = dΩ
cache = jac_cache[1]
test_affine_decomposition_lhs(cache,feop,fesolver,rbjact,meas,sols_test,params_test,sols,params)

# jcache,scache... = cache

# times = get_times(fesolver)
# ndofs_row = num_free_dofs(feop.test)
# ndofs_col = num_free_dofs(get_trial(feop)(nothing,nothing))
# setsize!(jcache,(ndofs_row,ndofs_col))

# red_idx = rbjact.integration_domain.idx
# red_times = rbjact.integration_domain.times
# red_meas = rbjact.integration_domain.meas

# #test_red_meas(feop,fesolver,sols,μ,red_meas)

# ode_op = get_algebraic_operator(feop)
# ode_cache = allocate_cache(ode_op,params_test,red_times)

# if length(red_times) < length(times)
#   A = get_array(jcache;len=length(red_times)*length(params_test))
#   time_idx = findall(x->x in red_times,times)
#   idx = param_time_idx(time_idx,length(params_test))
#   _sols = PTArray(sols_test[idx])
# else
#   A = get_array(jcache)
#   _sols = sols_test
# end
# Afull = copy(A)
# full_idx = findnz(Afull[1][:])[1]
# Jac = collect_jacobians!(A,fesolver,ode_op,_sols,params_test,red_times,ode_cache,red_idx,red_meas;i)
# Jac_full = collect_jacobians!(Afull,fesolver,ode_op,_sols,params_test,red_times,ode_cache,full_idx,meas;i)
# Jac_offline,_ = collect_jacobians_for_trian(fesolver,feop,sols[1:30],params[1:30],times;i)
# basis_space = tpod(Jac_offline[1])
# interp_idx_space = get_interpolation_idx(basis_space)
# err_jac = maximum(abs.(Jac-Jac_full[interp_idx_space,:]))

# coeff = mdeim_solve!(scache[1],rbjact.mdeim_interpolation,Jac)
# coeff_ok = basis_space'*Jac_full
# err_coeff = maximum(abs.(coeff-coeff_ok))
# println("Jacobian #$i coefficient difference for selected triangulation is $err_coeff")

# luok = lu(basis_space[interp_idx_space,:])
# luok.L ≈ rbjact.mdeim_interpolation.L

# _coeff = mdeim_solve!(scache[1],luok,Jac)
# _err_coeff = maximum(abs.(_coeff-coeff_ok))
