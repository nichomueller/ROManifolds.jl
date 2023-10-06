function test_sols()
  nparams = length(sols[1])
  ntimes = length(sols)
  @check all([!iszero(sols[nt][np]) for nt = 1:ntimes,np = 1:nparams])
  @check all([!iszero(sols_test[(nt-1)*nparams+np]) for nt = 1:ntimes,np = 1:nparams])
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

  b = get_array(rcache;len=length(red_times)*length(μ))
  sols = get_solutions_at_times(sols,fesolver,red_times)
  bfull = copy(b)
  res = collect_residuals_for_idx!(b,fesolver,sols,μ,red_times,red_idx,red_meas)
  res_full = collect_residuals_for_idx!(bfull,fesolver,sols,μ,red_times,full_idx,meas)
  res_offline,_ = collect_residuals_for_trian(fesolver,feop,offline_sols[1:30],offline_params[1:30],times)

  err_res = maximum(abs.(res-res_full[red_idx,:]))
  println("Residual difference for selected triangulation is $err_res")

  coeff = mdeim_solve!(scache[1],rbres.mdeim_interpolation,res)
  coeff_ok = try
    basis_space = tpod(res_offline[1])
    basis_space'*res_full
  catch
    basis_space = tpod(res_offline[2])
    basis_space'*res_full
  end
  err_coeff = maximum(abs.(coeff-coeff_ok))
  println("Residual coefficient difference for selected triangulation is $err_coeff")
  return coeff,coeff_ok
end

rbres,rbjac = rbrhs,rblhs
sols_test,params_test = load_test(info,feop,fesolver)
res_cache,jac_cache = allocate_sys_cache(feop,fesolver,rbspace,sols_test,params_test)

rbrest = rbres[Ω]
meas = dΩ
cache = res_cache[1]

coeff,coeff_ok = test_affine_decomposition_rhs(cache,feop,fesolver,rbrest,meas,sols_test,params_test,sols,params)

# rcache,scache... = cache
# times = get_times(fesolver)
# ndofs = num_free_dofs(feop.test)
# setsize!(rcache,(ndofs,))
# red_idx = rbrest.integration_domain.idx
# red_times = rbrest.integration_domain.times
# red_meas = rbrest.integration_domain.meas
# test_sols()
# b = get_array(rcache;len=length(red_times)*length(params_test))
# _sols = get_solutions_at_times(sols_test,fesolver,red_times)
# test_sols()
# Res = collect_residuals_for_idx!(b,fesolver,_sols,params_test,red_times,red_idx,red_meas)
# test_sols()

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

  A = get_array(jcache;len=length(red_times)*length(μ))
  sols = get_solutions_at_times(sols,fesolver,red_times)

  Afull = copy(A)
  full_idx = findnz(Afull[1][:])[1]
  jac = collect_jacobians_for_idx!(A,fesolver,sols,μ,red_times,red_idx,red_meas;i)
  jac_full = collect_jacobians_for_idx!(Afull,fesolver,sols,μ,red_times,full_idx,meas;i)
  jac_offline,_ = collect_jacobians_for_trian(fesolver,feop,offline_sols[1:30],offline_params[1:30],times;i)
  basis_space = tpod(jac_offline[1])
  interp_idx_space = get_interpolation_idx(basis_space)
  err_jac = maximum(abs.(jac-jac_full[interp_idx_space,:]))
  println("Jacobian #$i difference for selected triangulation is $err_jac")

  coeff = mdeim_solve!(scache[1],rbjac.mdeim_interpolation,jac)
  coeff_ok = basis_space'*jac_full
  err_coeff = maximum(abs.(coeff-coeff_ok))
  println("Jacobian #$i coefficient difference for selected triangulation is $err_coeff")
  return coeff,coeff_ok
end

i = 1
rbjact = rbjac[i][Ω]

meas = dΩ
cache = jac_cache[1]
coeff,coeff_ok = test_affine_decomposition_lhs(cache,feop,fesolver,rbjact,meas,sols_test,params_test,sols,params)

function test_rb_contribution_rhs(
  cache,
  feop::PTFEOperator,
  fesolver::PODESolver,
  ad::RBAffineDecomposition,
  rbspace::RBSpace,
  meas::Measure,
  sols::PTArray,
  params::Table;
  st_mdeim=true)

  coeff_cache,rb_cache = cache
  coeff = rhs_coefficient!(coeff_cache,feop,fesolver,ad,sols,params;st_mdeim)
  basis_space_proj = ad.basis_space
  basis_time = last(ad.basis_time)
  c1 = coeff[1]
  contrib1 = evaluate!(RBContributionMap(),rb_cache,basis_space_proj,basis_time,c1)

  rcache, = coeff_cache

  ndofs = num_free_dofs(feop.test)
  setsize!(rcache,(ndofs,))

  red_times = ad.integration_domain.times
  full_idx = collect(1:test.nfree)

  b = get_array(rcache;len=length(red_times)*length(params))
  sols = get_solutions_at_times(sols,fesolver,red_times)
  res_full = collect_residuals_for_idx!(b,fesolver,sols,params,red_times,full_idx,meas)
  contrib1_ok = space_time_projection(rbspace,res_full[:,1:get_time_ndofs(fesolver)])
  err_contrib = maximum(abs.(contrib1-contrib1_ok))
  println("Residual contribution difference for selected triangulation is $err_contrib")
  return contrib1,contrib1_ok
end

rbres,rbjac = rbrhs,rblhs
sols_test,params_test = load_test(info,feop,fesolver)
res_cache,jac_cache = allocate_sys_cache(feop,fesolver,rbspace,sols_test,params_test)

rbrest = rbres[Ω]
meas = dΩ

# contrib1,contrib1_ok = test_rb_contribution_rhs(res_cache,feop,fesolver,rbrest,rbspace,meas,sols_test,params_test;st_mdeim)

coeff_cache,rb_cache = res_cache
ptcoeff = rhs_coefficient!(coeff_cache,feop,fesolver,rbrest,sols_test,params_test;st_mdeim)
basis_space_proj = rbrest.basis_space
basis_time = last(rbrest.basis_time)
coeff = ptcoeff[1]
proj1 = testitem(proj_basis_space)

cache_coeff,cache_proj,cache_proj_global = rb_cache
num_rb_times = size(basis_time,2)
num_rb_dofs = length(proj1)*size(basis_time,2)
setsize!(cache_coeff,(num_rb_times,))
setsize!(cache_proj,(num_rb_dofs,))
setsize!(cache_proj_global,(num_rb_dofs,))

array_coeff = cache_coeff.array
array_proj = cache_proj.array
array_proj_global = cache_proj_global.array

@inbounds for i = axes(coeff,2)
  array_coeff .= basis_time'*coeff[:,i]
  LinearAlgebra.kron!(array_proj,proj_basis_space[i],array_coeff)
  array_proj_global .+= array_proj
end

array_proj_global
