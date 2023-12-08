function test_affine_decomposition_rhs(
  cache,
  op::PTOperator,
  op_offline::PTOperator,
  rbres::Vector{<:RBAffineDecomposition{T}},
  rbspace::RBSpace{T},) where T

  test = op.odeop.feop.test
  mdeim_cache,rb_cache = cache
  collect_cache,coeff_cache = mdeim_cache

  times = op.tθ
  nfree = length(get_free_dof_ids(test))

  res = collect_reduced_residuals!(collect_cache,op,rbres)
  res_full, = collect_residuals_for_trian(op)
  res_offline, = collect_residuals_for_trian(op_offline)

  for i in eachindex(res)
    rbres_i,res_i,res_full_i,res_offline_i = rbres[i],res[i],res_full[i],res_offline[i]
    coeff = rb_coefficient!(coeff_cache,rbres_i,res_i;st_mdeim=false)
    basis_space = tpod(res_offline_i)#tpod(recast(res_offline_i))
    space_idx = get_interpolation_idx(basis_space)
    err_res = maximum(abs.(res_i-res_full_i[space_idx,:]))
    println("Residual difference for selected triangulation is $err_res")

    for n = 1:length(op.μ)
      resn = res_full_i[:,(n-1)*length(times)+1:n*length(times)]
      coeff_ok = transpose(basis_space'*resn)
      coeffn = coeff[n]
      err_coeff = maximum(abs.(coeffn)-abs.(coeff_ok))
      println("Residual coefficient difference for selected triangulation is $err_coeff")
    end

    basis_space_proj = rbres_i.basis_space
    basis_time = last(rbres_i.basis_time)
    contribs = Vector{Vector{T}}(undef,length(coeff))
    k = RBVecContributionMap()
    @inbounds for i = eachindex(coeff)
      contribs[i] = copy(evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff[i]))
    end
    for n in eachindex(op.μ)
      tidx = (n-1)*length(times)+1 : n*length(times)
      nzmidx = NnzMatrix(Mabla.RB.Nonaffine(),res_full_i[:,tidx],res_full_i.nonzero_idx,nfree,1)
      contrib_ok = space_time_projection(nzmidx,rbspace)
      err_contrib = norm(contribs[n]-contrib_ok,Inf)
      println("Residual contribution difference for selected triangulation is $err_contrib")
    end
  end
end

function test_affine_decomposition_lhs(
  cache,
  op::PTOperator,
  op_offline::PTOperator,
  rbjac::Vector{<:RBAffineDecomposition{T}},
  rbspace_row::RBSpace{T},
  rbspace_col::RBSpace{T};
  j=1) where T

  test = op.odeop.feop.test
  mdeim_cache,rb_cache = cache
  collect_cache,coeff_cache = mdeim_cache
  combine_projections = (x,y) -> j == 1 ? θ*x+(1-θ)*y : θ*x-θ*y

  times = op.tθ
  nfree = length(get_free_dof_ids(test))

  jac = collect_reduced_jacobians!(collect_cache,op,rbjac;i=j)
  jac_full, = collect_jacobians_for_trian(op;i=j)
  jac_offline, = collect_jacobians_for_trian(op_offline;i=j)

  for i in eachindex(jac)
    rbjac_i,jac_i,jac_full_i,jac_offline_i = rbjac[i],jac[i],jac_full[i],jac_offline[i]
    coeff = rb_coefficient!(coeff_cache,rbjac_i,jac_i;st_mdeim=false)
    basis_space = tpod(jac_offline_i)

    for n = 1:length(op.μ)
      jacn = jac_full_i[:,(n-1)*length(times)+1:n*length(times)]
      coeff_ok = transpose(basis_space'*jacn)
      coeffn = coeff[n]
      err_coeff = maximum(abs.(coeffn)-abs.(coeff_ok))
      println("Jacobian #$j coefficient difference for selected triangulation is $err_coeff")
    end

    basis_space_proj = rbjac_i.basis_space
    basis_time = last(rbjac_i.basis_time)
    contribs = Vector{Matrix{T}}(undef,length(coeff))
    k = RBMatContributionMap()
    @inbounds for i = eachindex(coeff)
      contribs[i] = copy(evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff[i]))
    end
    for n in eachindex(op.μ)
      tidx = (n-1)*length(times)+1 : n*length(times)
      nzmidx = NnzMatrix(Mabla.RB.Nonaffine(),jac_full_i[:,tidx],jac_full_i.nonzero_idx,nfree,1)
      contrib_ok = space_time_projection(nzmidx,rbspace_row,rbspace_col;combine_projections)
      err_contrib = norm(contribs[n]-contrib_ok,Inf)
      println("Jacobian #$j contribution difference for selected triangulation is $err_contrib")
    end
  end
end

nsnaps_test = 10
snaps_train,params_train = sols[1:nsnaps_mdeim],params[1:nsnaps_mdeim]
snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
op = get_ptoperator(fesolver,feop,snaps_test,params_test)
op_offline = get_ptoperator(fesolver,feop,snaps_train,params_train)
(rhs_cache,lhs_cache),_ = Gridap.ODEs.TransientFETools.allocate_cache(op,rbspace)

test_affine_decomposition_rhs(rhs_cache,op,op_offline,rbrhs.affine_decompositions,rbspace)
j = 1
test_affine_decomposition_lhs(lhs_cache,op,op_offline,rblhs[j].affine_decompositions,rbspace,rbspace;j)
