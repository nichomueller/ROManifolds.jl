rbres,rbjac = rbrhs,rblhs
sols_test,params_test = load_test(info,feop,fesolver)
res_cache,jac_cache = allocate_online_cache(feop,fesolver,sols_test,params_test)

function test_rb_contribution_rhs(
  cache,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbres::BlockRBAlgebraicContribution,
  rbspace::BlockRBSpace,
  sols::Vector{<:PTArray},
  params::Table;
  kwargs...)

  nblocks = get_nblocks(rbres)
  vsols = vcat(sols...)
  for row = 1:nblocks
    touched_row = rbres.touched[row]
    if touched_row
      println("Block $row")
      feop_row = feop[row,:]
      rbres_row = rbres[row]
      rbspace_row = rbspace[row]
      test_rb_contribution_rhs(
        cache,feop_row,fesolver,rbres_row,rbspace_row,vsols,params;kwargs...)
    end
  end
end

test_rb_contribution_rhs(res_cache,feop,fesolver,rbres,rbspace,sols_test,params_test;st_mdeim)

function test_rb_contribution_lhs(
  cache,
  feop::PTFEOperator,
  fesolver::PODESolver,
  rbjac::BlockRBAlgebraicContribution,
  rbspace::BlockRBSpace,
  sols::Vector{<:PTArray},
  params::Table;
  kwargs...)

  nblocks = get_nblocks(rbjac)
  for (row,col) = index_pairs(nblocks,nblocks)
    touched_row_col = rbjac.touched[row,col]
    if touched_row_col
      println("Block ($row,$col)")
      feop_row = feop[row,col]
      rbjac_row_col = rbjac[row,col]
      sols_col = sols[col]
      rbspace_col = rbspace[col]
      test_rb_contribution_lhs(
        cache,feop_row,fesolver,rbjac_row_col,rbspace_col,sols_col,params;kwargs...)
    end
  end
end

i = 1
test_rb_contribution_lhs(jac_cache,feop,fesolver,rbjac[i],rbspace,sols_test,params_test;i,st_mdeim)


  #
### test_rb_contribution_rhs(res_cache,feop,fesolver,rbres,rbspace,sols_test,params_test;st_mdeim)
times = get_times(fesolver)
nblocks = get_nblocks(rbres)
vsols = vcat(sols_test...)
coeff_cache,rb_cache = res_cache
rcache, = coeff_cache
row = 1
feop_row = feop[row,:]
rbres_row = rbres[row]
rbspace_row = rbspace[row]
ndofs = num_free_dofs(feop_row.test)
setsize!(rcache,(ndofs,))
b = PTArray(map(x->x.array,rcache[1:length(times)*length(params_test)]))
# res_full = collect_residuals_for_idx!(b,fesolver,feop,vsols,params_test,times,collect(1:39),dΩ)

dtθ = θ == 0.0 ? dt : dt*θ
ode_op = get_algebraic_operator(feop_row)
ode_cache = allocate_cache(ode_op,params_test,times)
ode_cache = update_cache!(ode_cache,ode_op,params_test,times)
sols_cache = copy(vsols) .* 0.
nlop = get_nonlinear_operator(ode_op,params_test,times,dtθ,vsols,ode_cache,sols_cache)
ress = residual!(b,nlop,vsols)
