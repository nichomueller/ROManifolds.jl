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
  full_idx = collect(get_free_dof_ids(test))
  nfree = length(get_free_dof_ids(test))

  b = copy(first(collect_cache))
  sols = op.u0
  res = collect_reduced_residuals!(collect_cache,op,rbres)
  res_full, = residual_for_trian!(b,op,sols)
  res_offline, = collect_residuals_for_trian(op_offline)

  for i in eachindex(res)
    rbres_i,res_i,_res_full_i,res_offline_i = rbres[i],res[i],res_full[i],res_offline[i]
    res_full_i = NnzMatrix(_res_full_i)
    red_idx = rbres_i.integration_domain.idx
    err_res = maximum(abs.(res_i-res_full_i[red_idx,:]))
    println("Residual difference for selected triangulation is $err_res")
    coeff = rb_coefficient!(coeff_cache,rbres_i,res_i;st_mdeim=false)
    basis_space = tpod(recast(res_offline_i))

    for n = 1:length(op_offline.μ)
      resn = res_full_i[:,(n-1)*length(times)+1:n*length(times)]
      coeff_ok = transpose(basis_space'*resn)
      coeffn = coeff[n]
      err_coeff = maximum(abs.(coeffn)-abs.(coeff_ok))
      println("Residual coefficient difference for selected triangulation is $err_coeff")
    end

    basis_space_proj = rbres_i.basis_space
    basis_time = last(rbres_i.basis_time)
    contribs = Vector{Vector{T}}(undef,length(coeff))
    k = RBVecContributionMap(T)
    @inbounds for i = eachindex(coeff)
      contribs[i] = copy(evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff[i]))
    end
    for n in eachindex(op_offline.μ)
      tidx = (n-1)*length(times)+1 : n*length(times)
      nzmidx = NnzMatrix{Nonaffine}(res_full_i[:,tidx],full_idx,nfree,1)
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
  full_idx = collect(get_free_dof_ids(test))
  nfree = length(get_free_dof_ids(test))

  b = copy(first(collect_cache))
  sols = op.u0
  jac = collect_reduced_jacobians!(collect_cache,op,rbjac;i=j)
  jac_full, = jacobian_for_trian!(b,op,sols,j)
  jac_offline, = collect_jacobians_for_trian(op_offline;i=j)

  for i in eachindex(jac)
    rbjac_i,jac_i,_jac_full_i,jac_offline_i = rbjac[i],jac[i],jac_full[i],jac_offline[i]
    jac_full_i = NnzMatrix(_jac_full_i)
    red_idx = rbjac_i.integration_domain.idx
    err_jac = maximum(abs.(jac_i-jac_full_i[red_idx,:]))
    println("Jacobian #$j difference for selected triangulation is $err_jac")
    coeff = rb_coefficient!(coeff_cache,rbjac_i,jac_i;st_mdeim=false)
    basis_space = tpod(jac_offline_i)

    for n = 1:length(op_offline.μ)
      jacn = jac_full_i[:,(n-1)*length(times)+1:n*length(times)]
      coeff_ok = transpose(basis_space'*jacn)
      coeffn = coeff[n]
      err_coeff = maximum(abs.(coeffn)-abs.(coeff_ok))
      println("Jacobian #$j coefficient difference for selected triangulation is $err_coeff")
    end

    basis_space_proj = rbjac_i.basis_space
    basis_time = last(rbjac_i.basis_time)
    contribs = Vector{Matrix{T}}(undef,length(coeff))
    k = RBMatContributionMap(T)
    @inbounds for i = eachindex(coeff)
      contribs[i] = copy(evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff[i]))
    end
    for n in eachindex(op_offline.μ)
      tidx = (n-1)*length(times)+1 : n*length(times)
      nzmidx = NnzMatrix{Nonaffine}(jac_full_i[:,tidx],full_idx,nfree,1)
      contrib_ok = space_time_projection(nzmidx,rbspace_row,rbspace_col;combine_projections)
      err_contrib = norm(contribs[n]-contrib_ok,Inf)
      println("Jacobian #$j contribution difference for selected triangulation is $err_contrib")
    end
  end
end

# function test_affine_decomposition_lhs(
#   cache,
#   op::PTOperator,
#   op_offline::PTOperator,
#   rbjact::RBAffineDecomposition,
#   meas::Measure,
#   rbspace_row::RBSpace{T},
#   rbspace_col::RBSpace{T};
#   i=1) where T

#   mdeim_cache,rb_cache = cache
#   A,scache... = mdeim_cache

#   times = op.tθ
#   red_idx = rbjact.integration_domain.idx
#   red_times = rbjact.integration_domain.times
#   red_meas = rbjact.integration_domain.meas
#   @assert red_times == times
#   combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : θ*x-θ*y

#   Afull = copy(A)
#   sols = op.u0
#   full_idx = findnz(Afull[1][:])[1]
#   nfree = length(get_free_dof_ids(op.odeop.feop.test))
#   jac = collect_reduced_jacobians!(A,op,sols,red_idx,red_meas;i)
#   jac_full = collect_reduced_jacobians!(Afull,op,sols,full_idx,red_meas;i)
#   jac_offline,trian = collect_jacobians_for_trian(op_offline;i)
#   basis_space = tpod(jac_offline[1])
#   interp_idx_space = get_interpolation_idx(basis_space)
#   err_jac = maximum(abs.(jac-jac_full[interp_idx_space,:]))
#   println("jacobian #$i difference for selected triangulation is $err_jac")

#   idx = get_idx_same_trian(trian,meas)
#   coeff = mdeim_solve!(scache,rbjact,jac;st_mdeim=false)
#   basis_space = tpod(jac_offline[idx])
#   for n = 1:length(op_offline.μ)
#     jacn = jac_full[:,(n-1)*length(times)+1:n*length(times)]
#     coeff_ok = transpose(basis_space'*jacn)
#     coeffn = coeff[n]
#     println(length(coeff))
#     println(size(coeffn))
#     err_coeff = maximum(abs.(coeffn)-abs.(coeff_ok))
#     println("jacobian coefficient difference for selected triangulation is $err_coeff")
#   end

#   basis_space_proj = rbjact.basis_space
#   basis_time = last(rbjact.basis_time)
#   contribs = Vector{Matrix{T}}(undef,length(coeff))
#   k = RBMatContributionMap(T)
#   @inbounds for i = eachindex(coeff)
#     contribs[i] = copy(evaluate!(k,rb_cache,basis_space_proj,basis_time,coeff[i]))
#   end
#   for n in eachindex(op_offline.μ)
#     tidx = (n-1)*length(times)+1 : n*length(times)
#     nzmidx = NnzMatrix(jac_full[:,tidx],full_idx,nfree,1)
#     contrib_ok = space_time_projection(nzmidx,rbspace_row,rbspace_col;combine_projections)
#     err_contrib = norm(contribs[n]-contrib_ok,Inf)
#     println("jacobian #$i contribution difference for selected triangulation is $err_contrib")
#   end
# end

nsnaps_test = 10
snaps_train,params_train = sols[1:nsnaps_test],params[1:nsnaps_test]
snaps_test,params_test = sols[end-nsnaps_test+1:end],params[end-nsnaps_test+1:end]
op = get_ptoperator(fesolver,feop,snaps_test,params_test)
op_offline = get_ptoperator(fesolver,feop,snaps_train,params_train)
rhs_cache,lhs_cache = allocate_cache(op,snaps_test)

test_affine_decomposition_rhs(rhs_cache,op,op_offline,rbrhs.affine_decompositions,rbspace)
j = 1
test_affine_decomposition_lhs(lhs_cache,op,op_offline,rblhs[i].affine_decompositions,rbspace,rbspace;j)
