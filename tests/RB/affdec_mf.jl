include("affdec_sf.jl")

function test_affine_decomposition_rhs(
  cache,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbres::BlockRBAlgebraicContribution,
  sols_test::Vector{<:PTArray},
  params_test::Table,
  sols::Vector{<:PTArray},
  params::Table;
  kwargs...)

  nblocks = get_nblocks(rbres)
  offsets = field_offsets(feop.test)
  vsols = vcat(sols...)
  vsols_test = vcat(sols_test...)
  for row = 1:nblocks
    cache_row = cache_at_index(cache,offsets[row]+1:offsets[row+1])
    touched_row = rbres.touched[row]
    if touched_row
      println("Block $row")
      feop_row = feop[row,:]
      rbres_row = rbres[row]
      test_affine_decomposition_rhs(
        cache_row[1],feop_row,fesolver,rbres_row,vsols_test,params_test,vsols,params;kwargs...)
    end
  end
end

function test_affine_decomposition_lhs(
  cache,
  feop::PTFEOperator,
  fesolver::PThetaMethod,
  rbjac::BlockRBAlgebraicContribution,
  sols_test::Vector{<:PTArray},
  params_test::Table,
  sols::Vector{<:PTArray},
  params::Table;
  kwargs...)

  nblocks = get_nblocks(rbjac)
  offsets = field_offsets(feop.test)
  for (row,col) = index_pairs(nblocks,nblocks)
    cache_row_col = cache_at_index(cache,offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1])
    touched_row_col = rbjac.touched[row,col]
    if touched_row_col
      println("Block $row")
      feop_row_col = feop[row,col]
      rbjac_row_col = rbjac[row,col]
      sols_col = sols[col]
      sols_test_col = sols_test[col]
      test_affine_decomposition_lhs(
        cache_row_col,feop_row_col,fesolver,rbjac_row_col,sols_test_col,params_test,sols_col,params;kwargs...)
    end
  end
end

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
  offsets = field_offsets(feop.test)
  vsols = vcat(sols...)
  for row = 1:nblocks
    cache_row = cache_at_index(cache,offsets[row]+1:offsets[row+1])
    touched_row = rbres.touched[row]
    if touched_row
      println("Block $row")
      feop_row = feop[row,:]
      rbres_row = rbres[row]
      rbspace_row = rbspace[row]
      test_rb_contribution_rhs(
        cache_row,feop_row,fesolver,rbres_row,rbspace_row,vsols,params;kwargs...)
    end
  end
end

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
  offsets = field_offsets(feop.test)
  for (row,col) = index_pairs(nblocks,nblocks)
    touched_row_col = rbjac.touched[row,col]
    if touched_row_col
      println("Block ($row,$col)")
      cache_row_col = cache_at_index(cache,offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1])
      feop_row_col = feop[row,col]
      rbjac_row_col = rbjac[row,col]
      sols_col = sols[col]
      rbspace_row = rbspace[row]
      rbspace_col = rbspace[col]
      test_rb_contribution_lhs(
        cache_row_col,feop_row_col,fesolver,rbjac_row_col,rbspace_row,rbspace_col,sols_col,params;kwargs...)
    end
  end
end

xn,μn = [PTArray(snaps_test[1][1:ntimes]),PTArray(snaps_test[2][1:ntimes])],params_test[1:1]
rhs_cache,lhs_cache = allocate_online_cache(feop,fesolver,xn,μn)

test_affine_decomposition_rhs(
  res_cache,feop,fesolver,rbres,xn,μn,sols[1:nsnaps_system],params[1:nsnaps_system];st_mdeim)
test_rb_contribution_rhs(res_cache,feop,fesolver,rbres,rbspace,sols_test,params_test;st_mdeim)

i = 1
test_rb_contribution_lhs(jac_cache,feop,fesolver,rbjac[i],rbspace,sols_test,params_test;i,st_mdeim)
