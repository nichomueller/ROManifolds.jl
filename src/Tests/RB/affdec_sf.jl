rbres,rbjac = rbrhs,rblhs
sols_test,params_test = load_test(info,feop,fesolver)
res_cache,jac_cache = allocate_sys_cache(feop,fesolver,rbspace,sols_test,params_test)

function test_affine_decomposition_rhs(
  cache,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbrest::RBAffineDecomposition,
  meas::Measure,
  sols_test::PTArray,
  params_test::Table,
  sols::Snapshots,
  params::Table)

  nsnaps_system = 50
  rcache,scache... = cache

  times = get_times(fesolver)
  ndofs = num_free_dofs(feop.test)
  setsize!(rcache,(ndofs,))

  red_idx = rbrest.integration_domain.idx
  red_times = rbrest.integration_domain.times
  red_meas = rbrest.integration_domain.meas
  full_idx = collect(1:test.nfree)

  b = get_array(rcache;len=length(red_times)*length(params_test))
  sols_test = get_solutions_at_times(sols_test,fesolver,red_times)
  bfull = copy(b)
  Res = collect_residuals_for_idx!(b,fesolver,sols_test,params_test,red_times,red_idx,red_meas)
  Res_full = collect_residuals_for_idx!(bfull,fesolver,sols_test,params_test,red_times,full_idx,meas)
  Res_offline,trian = collect_residuals_for_trian(
      fesolver,feop,sols[1:nsnaps_system],params[1:nsnaps_system],times)

  err_res = maximum(abs.(Res-Res_full[red_idx,:]))
  println("Residual difference for selected triangulation is $err_res")

  function ret_idx()
    _trian = get_triangulation(meas)
    for (it,t) in enumerate(trian)
      if t == _trian
        return it
        break
      end
    end
    @unreachable
  end

  idx = ret_idx()
  coeff = mdeim_solve!(scache[1],rbrest.mdeim_interpolation,Res)
  basis_space = tpod(recast(Res_offline[idx]))
  coeff_ok = basis_space'*Res_full
  err_coeff = maximum(abs.(coeff)-abs.(coeff_ok))
  println("Residual coefficient difference for selected triangulation is $err_coeff")
  return coeff,coeff_ok
end

rbrest = rbres[Ω]
meas = dΩ
cache = res_cache[1]

# offline error
coeff,coeff_ok = test_affine_decomposition_rhs(cache,feop,fesolver,rbrest,meas,sols[1],params[1:1],sols,params)
# online error
coeff,coeff_ok = test_affine_decomposition_rhs(cache,feop,fesolver,rbrest,meas,sols_test,params_test,sols,params)

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
  jac_offline,_ = collect_jacobians_for_trian(
    fesolver,feop,offline_sols[1:nsnaps_system],offline_params[1:nsnaps_system],times;i)
  basis_space = tpod(jac_offline[1])
  interp_idx_space = get_interpolation_idx(basis_space)
  err_jac = maximum(abs.(jac-jac_full[interp_idx_space,:]))
  println("Jacobian #$i difference for selected triangulation is $err_jac")

  coeff = mdeim_solve!(scache[1],rbjac.mdeim_interpolation,jac)
  coeff_ok = basis_space'*jac_full
  err_coeff = maximum(abs.(coeff)-abs.(coeff_ok))
  println("Jacobian #$i coefficient difference for selected triangulation is $err_coeff")
  return coeff,coeff_ok
end

i = 1
rbjact = rbjac[i][Ω]

meas = dΩ
cache = jac_cache[1]

# offline error
coeff,coeff_ok = test_affine_decomposition_lhs(cache,feop,fesolver,rbjact,meas,sols[1],params[1:1],sols,params;i)
# online error
coeff,coeff_ok = test_affine_decomposition_lhs(cache,feop,fesolver,rbjact,meas,sols_test,params_test,sols,params;i)

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
  contribs = Vector{Vector{Float}}(undef,length(coeff))
  @inbounds for i = eachindex(coeff)
    contribs[i] = copy(evaluate!(RBContributionMap(),rb_cache,basis_space_proj,basis_time,coeff[i]))
  end

  rcache, = coeff_cache

  ndofs = num_free_dofs(feop.test)
  setsize!(rcache,(ndofs,))

  red_times = ad.integration_domain.times
  full_idx = collect(1:test.nfree)

  b = get_array(rcache;len=length(red_times)*length(params))
  sols = get_solutions_at_times(sols,fesolver,red_times)
  res_full = collect_residuals_for_idx!(b,fesolver,sols,params,red_times,full_idx,meas)
  for n in eachindex(params)
    tidx = (n-1)*length(red_times)+1 : n*length(red_times)
    contrib_ok = space_time_projection(res_full[:,tidx],rbspace)
    err_contrib = ℓ∞(contribs[n]-contrib_ok)
    println("Residual contribution difference for selected triangulation is $err_contrib")
  end
end

rbrest = rbres[Γn]
meas = dΓn

test_rb_contribution_rhs(res_cache,feop,fesolver,rbrest,rbspace,meas,sols_test,params_test;st_mdeim)

function test_rb_contribution_lhs(
  cache,
  feop::PTFEOperator,
  fesolver::PODESolver,
  ad::RBAffineDecomposition,
  rbspace::RBSpace,
  meas::Measure,
  sols::PTArray,
  params::Table;
  st_mdeim=true,
  i=1)

  combine_projections = (x,y) -> i == 1 ? fesolver.θ*x+(1-fesolver.θ)*y : x-y
  coeff_cache,rb_cache = cache
  coeff = lhs_coefficient!(coeff_cache,feop,fesolver,ad,sols,params;st_mdeim,i)
  basis_space_proj = ad.basis_space
  basis_time = last(ad.basis_time)
  contribs = Vector{Matrix{Float}}(undef,length(coeff))
  @inbounds for i = eachindex(coeff)
    contribs[i] = copy(evaluate!(RBContributionMap(),rb_cache,basis_space_proj,basis_time,coeff[i]))
  end

  jcache, = coeff_cache

  ndofs_row = num_free_dofs(feop.test)
  ndofs_col = num_free_dofs(get_trial(feop)(nothing,nothing))
  setsize!(jcache,(ndofs_row,ndofs_col))

  red_times = ad.integration_domain.times

  A = get_array(jcache;len=length(red_times)*length(params))
  Afull = copy(A)
  full_idx = findnz(Afull[1][:])[1]
  jac_full = collect_jacobians_for_idx!(A,fesolver,sols,params,red_times,full_idx,meas;i)

  sols = get_solutions_at_times(sols,fesolver,red_times)
  for n in eachindex(params)
    tidx = (n-1)*length(red_times)+1 : n*length(red_times)
    nzmidx = NnzMatrix(jac_full[:,tidx],findnz(Afull[1][:])[1],test.nfree,1)
    contrib_ok = space_time_projection(nzmidx,rbspace,rbspace;combine_projections)
    err_contrib = ℓ∞(contribs[n]-contrib_ok)
    println("Jacobian #$i contribution difference for selected triangulation is $err_contrib")
  end
end

i = 1
rbjact = rbjac[i][Ω]

meas = dΩ
test_rb_contribution_lhs(jac_cache,feop,fesolver,rbjact,rbspace,meas,sols_test,params_test;i,st_mdeim)

# NONAFFINE CASE
i = 1
rbjact = rbjac[i][Ω]
meas = dΩ
coeff_cache,rb_cache = jac_cache
coeff = lhs_coefficient!(coeff_cache,feop,fesolver,rbjact,sols_test,params_test;st_mdeim,i)
basis_space_proj = rbjact.basis_space
basis_time = last(rbjact.basis_time)
idx = 1
cidx = coeff[idx]
contrib = evaluate!(RBContributionMap(),rb_cache,basis_space_proj,basis_time,cidx)

times = get_times(fesolver)
ntimes = get_time_ndofs(fesolver)
p,t = params[slow_idx(idx,ntimes)],times[fast_idx(idx,ntimes)]
A = assemble_matrix((du,dv)->∫(a(p,t)*∇(dv)⋅∇(du))dΩ,trial(p,t),test)
Ared = rbspace.basis_space'*A*rbspace.basis_space
btbt = rbspace.basis_time'*rbspace.basis_time
btbt_shift = rbspace.basis_time[2:end,:]'*rbspace.basis_time[1:end-1,:]
Arb = LinearAlgebra.kron(Ared,θ*btbt+(1-θ)*btbt_shift)

# AFFINE CASE WORKS
i = 2
rbjact = rbjac[i][Ω]
meas = dΩ
coeff_cache,rb_cache = jac_cache
coeff = lhs_coefficient!(coeff_cache,feop,fesolver,rbjact,sols_test,params_test;st_mdeim,i)
basis_space_proj = rbjact.basis_space
basis_time = last(rbjact.basis_time)
idx = 1
cidx = coeff[idx]
contrib = evaluate!(RBContributionMap(),rb_cache,basis_space_proj,basis_time,cidx)

all_contribs = map(coeff) do cn
  evaluate!(RBContributionMap(),rb_cache,basis_space_proj,basis_time,cn)
end
@check all([ci ≈ contrib for ci = all_contribs.array])

M = assemble_matrix((du,dv)->∫(dv*du)dΩ,trial(params[1],dt),test)/(dt*θ)
Mred = rbspace.basis_space'*M*rbspace.basis_space
btbt = rbspace.basis_time'*rbspace.basis_time
btbt_shift = rbspace.basis_time[2:end,:]'*rbspace.basis_time[1:end-1,:]
Mrb = LinearAlgebra.kron(Mred,btbt-btbt_shift)
@check isapprox(Mrb,contrib)
