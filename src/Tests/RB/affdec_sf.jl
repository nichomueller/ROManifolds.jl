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

for f in (:test_affine_decomposition_rhs,:test_affine_decomposition_lhs)
  @eval begin
    function $f(
      cache,
      op::PTAlgebraicOperator,
      op_offline::PTAlgebraicOperator,
      ad::RBAlgebraicContribution,
      args...;
      kwargs...)

      for trian in get_domains(ad)
        meas = get_measure(feop,trian)
        $f(cache,op,op_offline,ad[trian],meas,args...;kwargs...)
      end
    end
  end
end

function test_affine_decomposition_rhs(
  cache,
  op::PTAlgebraicOperator,
  op_offline::PTAlgebraicOperator,
  rbrest::RBAffineDecomposition{T},
  meas::Measure,
  rbspace::RBSpace{T},) where T

  coeff_cache,rb_cache = cache
  b,scache... = coeff_cache

  times = op.tθ
  red_idx = rbrest.integration_domain.idx
  red_times = rbrest.integration_domain.times
  @assert red_times == times
  red_meas = rbrest.integration_domain.meas
  full_idx = collect(get_free_dof_ids(op.odeop.feop.test))
  nfree = length(get_free_dof_ids(op.odeop.feop.test))

  bfull = copy(b)
  sols = op.u0
  res = collect_residuals_for_idx!(b,op,sols,red_idx,red_meas)
  res_full = collect_residuals_for_idx!(bfull,op,sols,full_idx,red_meas)

  err_res = maximum(abs.(res-res_full[red_idx,:]))
  println("Residual difference for selected triangulation is $err_res")

  res_offline,trian = collect_residuals_for_trian(op_offline)
  idx = get_idx_same_trian(trian,meas)
  coeff = mdeim_solve!(scache,rbrest,res;st_mdeim=false)
  basis_space = tpod(recast(res_offline[idx]))
  for n = 1:length(op_offline.μ)
    resn = res_full[:,(n-1)*length(times)+1:n*length(times)]
    coeff_ok = transpose(basis_space'*resn)
    coeffn = coeff[n]
    err_coeff = maximum(abs.(coeffn)-abs.(coeff_ok))
    println("Residual coefficient difference for selected triangulation is $err_coeff")
  end

  basis_space_proj = rbrest.basis_space
  basis_time = last(rbrest.basis_time)
  contribs = Vector{Vector{T}}(undef,length(coeff))
  k = RBVecContributionMap(T)
  @inbounds for i = eachindex(coeff)
    contribs[i] = copy(evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff[i]))
  end
  for n in eachindex(op_offline.μ)
    tidx = (n-1)*length(red_times)+1 : n*length(red_times)
    nzmidx = NnzMatrix(res_full[:,tidx],full_idx,nfree,1)
    contrib_ok = space_time_projection(nzmidx,rbspace)
    err_contrib = norm(contribs[n]-contrib_ok,Inf)
    println("Residual contribution difference for selected triangulation is $err_contrib")
  end
end

function test_affine_decomposition_lhs(
  cache,
  op::PTAlgebraicOperator,
  op_offline::PTAlgebraicOperator,
  rbjact::RBAffineDecomposition,
  meas::Measure,
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T};
  i=1) where T

  coeff_cache,rb_cache = cache
  A,scache... = coeff_cache

  times = op.tθ
  red_idx = rbjact.integration_domain.idx
  red_times = rbjact.integration_domain.times
  red_meas = rbjact.integration_domain.meas
  @assert red_times == times
  combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y

  Afull = copy(A)
  sols = op.u0
  full_idx = findnz(Afull[1][:])[1]
  nfree = length(get_free_dof_ids(op.odeop.feop.test))
  jac = collect_jacobians_for_idx!(A,op,sols,red_idx,red_meas;i)
  jac_full = collect_jacobians_for_idx!(Afull,op,sols,full_idx,red_meas;i)
  jac_offline,trian = collect_jacobians_for_trian(op_offline;i)
  basis_space = tpod(jac_offline[1])
  interp_idx_space = get_interpolation_idx(basis_space)
  err_jac = maximum(abs.(jac-jac_full[interp_idx_space,:]))
  println("jacobian #$i difference for selected triangulation is $err_jac")

  idx = get_idx_same_trian(trian,meas)
  coeff = mdeim_solve!(scache,rbjact,jac;st_mdeim=false)
  basis_space = tpod(jac_offline[idx])
  for n = 1:length(op_offline.μ)
    jacn = jac_full[:,(n-1)*length(times)+1:n*length(times)]
    coeff_ok = transpose(basis_space'*jacn)
    coeffn = coeff[n]
    println(length(coeff))
    println(size(coeffn))
    err_coeff = maximum(abs.(coeffn)-abs.(coeff_ok))
    println("jacobian coefficient difference for selected triangulation is $err_coeff")
  end

  basis_space_proj = rbjact.basis_space
  basis_time = last(rbjact.basis_time)
  contribs = Vector{Matrix{T}}(undef,length(coeff))
  k = RBMatContributionMap(T)
  @inbounds for i = eachindex(coeff)
    contribs[i] = copy(evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff[i]))
  end
  for n in eachindex(op_offline.μ)
    tidx = (n-1)*length(red_times)+1 : n*length(red_times)
    nzmidx = NnzMatrix(jac_full[:,tidx],full_idx,nfree,1)
    contrib_ok = space_time_projection(nzmidx,rbspace_row,rbspace_col;combine_projections)
    err_contrib = norm(contribs[n]-contrib_ok,Inf)
    println("jacobian #$i contribution difference for selected triangulation is $err_contrib")
  end
end

nsnaps_test = 10
snaps_train,params_train = sols[1:nsnaps_test],params[1:nsnaps_test]
snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
op = get_ptoperator(fesolver,feop,snaps_test,params_test)
op_offline = get_ptoperator(fesolver,feop,snaps_train,params_train)
rhs_cache,lhs_cache = allocate_cache(op,snaps_test)

test_affine_decomposition_rhs(rhs_cache,op,op_offline,rbrhs,rbspace)
i = 2
test_affine_decomposition_lhs(lhs_cache,op,op_offline,rblhs[i],rbspace,rbspace;i)
