function get_idx_same_trian(trian,meas)
  _trian = get_triangulation(meas)
  for (it,t) in enumerate(trian)
    if t == _trian
      return it
      break
    end
  end
  @unreachable
end

for f in (:test_affine_decomposition_rhs,:test_affine_decomposition_lhs,
          :test_rb_contribution_rhs,:test_rb_contribution_lhs)
  @eval begin
    function $f(
      cache,
      feop::PTFEOperator,
      fesolver::PThetaMethod,
      rbres::RBAlgebraicContribution,
      args...;
      kwargs...)

      for trian in get_domains(rbres)
        meas = get_measure(feop,trian)
        rbrest = rbres[trian]
        $f(cache,feop,fesolver,rbrest,meas,args...;kwargs...)
      end
    end
  end
end

function test_affine_decomposition_rhs(
  cache,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbrest::RBAffineDecomposition,
  meas::Measure,
  sols_test::PTArray,
  params_test::Table,
  sols::PTArray,
  params::Table;
  st_mdeim=false)

  rcache,scache... = cache

  times = get_times(fesolver)
  red_idx = rbrest.integration_domain.idx
  red_times = rbrest.integration_domain.times
  red_meas = rbrest.integration_domain.meas
  full_idx = collect(get_free_dof_ids(feop.test))

  b = PTArray(rcache[1:length(red_times)*length(params_test)])
  sols_test = get_solutions_at_times(sols_test,fesolver,red_times)
  bfull = copy(b)
  Res = collect_residuals_for_idx!(b,fesolver,feop,sols_test,params_test,red_times,red_idx,red_meas)
  Res_full = collect_residuals_for_idx!(bfull,fesolver,feop,sols_test,params_test,red_times,full_idx,meas)
  Res_offline,trian = collect_residuals_for_trian(
      fesolver,feop,sols,params,times)

  err_res = maximum(abs.(Res-Res_full[red_idx,:]))
  println("Residual difference for selected triangulation is $err_res")

  idx = get_idx_same_trian(trian,meas)
  coeff = mdeim_solve!(scache,rbrest,Res;st_mdeim)
  basis_space = tpod(recast(Res_offline[idx]))
  for n = 1:length(params_test)
    Resn = Res_full[:,(n-1)*length(times)+1:n*length(times)]
    coeff_ok = transpose(basis_space'*Resn)
    coeffn = coeff[n]
    err_coeff = maximum(abs.(coeffn)-abs.(coeff_ok))
    println("Residual coefficient difference for selected triangulation is $err_coeff")
  end
end

function test_affine_decomposition_lhs(
  cache,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbjact::RBAffineDecomposition,
  meas::Measure,
  sols_test::PTArray,
  params_test::Table,
  sols::PTArray,
  params::Table;
  st_mdeim=false,
  i=1)

  jcache,scache... = cache

  times = get_times(fesolver)
  red_idx = rbjact.integration_domain.idx
  red_times = rbjact.integration_domain.times
  red_meas = rbjact.integration_domain.meas

  A = PTArray(jcache[1:length(red_times)*length(params_test)])
  sols_test = get_solutions_at_times(sols_test,fesolver,red_times)

  Afull = copy(A)
  full_idx = findnz(Afull[1][:])[1]
  Jac = collect_jacobians_for_idx!(A,fesolver,feop,sols_test,params_test,red_times,red_idx,red_meas;i)
  Jac_full = collect_jacobians_for_idx!(Afull,fesolver,feop,sols_test,params_test,red_times,full_idx,meas;i)
  Jac_offline,_ = collect_jacobians_for_trian(
    fesolver,feop,sols,params,times;i)
  basis_space = tpod(Jac_offline[1])
  interp_idx_space = get_interpolation_idx(basis_space)
  err_jac = maximum(abs.(Jac-Jac_full[interp_idx_space,:]))
  println("Jacobian #$i difference for selected triangulation is $err_jac")

  idx = get_idx_same_trian(trian,meas)
  coeff = mdeim_solve!(scache,rbjact,Jac;st_mdeim)
  basis_space = tpod(Jac_offline[idx])
  for n = 1:length(params_test)
    jacn = Jac_full[:,(n-1)*length(times)+1:n*length(times)]
    coeff_ok = transpose(basis_space'*jacn)
    coeffn = coeff[n]
    println(length(coeff))
    println(size(coeffn))
    err_coeff = maximum(abs.(coeffn)-abs.(coeff_ok))
    println("Jacobian coefficient difference for selected triangulation is $err_coeff")
  end
end

function test_rb_contribution_rhs(
  cache,
  feop::PTFEOperator,
  fesolver::PODESolver,
  ad::RBAffineDecomposition,
  meas::Measure,
  rbspace::RBSpace{T},
  sols::PTArray,
  params::Table;
  st_mdeim=true) where T

  coeff_cache,rb_cache = cache
  coeff = rhs_coefficient!(coeff_cache,feop,fesolver,ad,sols,params;st_mdeim)
  basis_space_proj = ad.basis_space
  basis_time = last(ad.basis_time)
  contribs = Vector{Vector{T}}(undef,length(coeff))
  k = RBVecContributionMap(T)
  @inbounds for i = eachindex(coeff)
    contribs[i] = copy(evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff[i]))
  end

  rcache, = coeff_cache
  red_times = ad.integration_domain.times
  full_idx = collect(get_free_dof_ids(feop.test))

  b = PTArray(rcache[1:length(red_times)*length(params)])
  sols = get_solutions_at_times(sols,fesolver,red_times)
  nfree = length(get_free_dof_ids(feop.test))
  res_full = collect_residuals_for_idx!(b,fesolver,feop,sols,params,red_times,full_idx,meas)
  global contrib_ok
  for n in eachindex(params)
    tidx = (n-1)*length(red_times)+1 : n*length(red_times)
    nzmidx = NnzMatrix(res_full[:,tidx],full_idx,nfree,1)
    contrib_ok = space_time_projection(nzmidx,rbspace)
    err_contrib = ℓ∞(contribs[n]-contrib_ok)
    println("Residual contribution difference for selected triangulation is $err_contrib")
  end
  return contribs,contrib_ok
end

function test_rb_contribution_lhs(
  cache,
  feop::PTFEOperator,
  fesolver::PODESolver,
  ad::RBAffineDecomposition,
  meas::Measure,
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T},
  sols::PTArray,
  params::Table;
  st_mdeim=true,
  i=1) where T

  combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y
  coeff_cache,rb_cache = cache
  coeff = lhs_coefficient!(coeff_cache,feop,fesolver,ad,sols,params;st_mdeim,i)
  basis_space_proj = ad.basis_space
  basis_time = last(ad.basis_time)
  contribs = Vector{Matrix{T}}(undef,length(coeff))
  k = RBMatContributionMap(T)
  @inbounds for i = eachindex(coeff)
    contribs[i] = copy(evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff[i]))
  end

  jcache, = coeff_cache
  red_times = ad.integration_domain.times

  A = PTArray(jcache[1:length(red_times)*length(params)])
  Afull = copy(A)
  full_idx = Afull[1][:].nzind
  jac_full = collect_jacobians_for_idx!(A,fesolver,feop,sols,params,red_times,full_idx,meas;i)

  sols = get_solutions_at_times(sols,fesolver,red_times)
  nfree = length(get_free_dof_ids(feop.test))
  global contrib_ok
  for n in eachindex(params)
    tidx = (n-1)*length(red_times)+1 : n*length(red_times)
    nzmidx = NnzMatrix(jac_full[:,tidx],full_idx,nfree,1)
    contrib_ok = space_time_projection(nzmidx,rbspace_row,rbspace_col;combine_projections)
    err_contrib = ℓ∞(contribs[n]-contrib_ok)
    println("Jacobian #$i contribution difference for selected triangulation is $err_contrib")
  end
  return contribs,contrib_ok
end

rbres,rbjac = rbrhs,rblhs
sols_test,params_test = load_test(info,feop,fesolver)
res_cache,jac_cache = allocate_online_cache(feop,fesolver,sols_test,params_test)

# test_affine_decomposition_rhs(
#   res_cache[1],feop,fesolver,rbres,sols_test,params_test,sols[1:nsnaps_system],params[1:nsnaps_system];st_mdeim)

# test_rb_contribution_rhs(
#   res_cache,feop,fesolver,rbres,rbspace,sols_test,params_test;st_mdeim)

# i = 1

# coeff,coeff_ok = test_affine_decomposition_lhs(
#   jac_cache[1],feop,fesolver,rbjac[i],sols_test,params_test,sols[1:nsnaps_system],params[1:nsnaps_system];i,st_mdeim)

# test_rb_contribution_lhs(
#   jac_cache,feop,fesolver,rbjac[i],rbspace,rbspace,sols_test,params_test;i,st_mdeim)
