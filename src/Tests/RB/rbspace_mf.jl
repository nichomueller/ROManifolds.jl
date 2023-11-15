μ = realization(feop,10)
sols, = collect_solutions(fesolver,feop,get_trial(feop),μ)
rbspace = reduced_basis(rbinfo,feop,sols,params)

norm_style = rbinfo.norm_style
nblocks = get_nblocks(sols)
blocks = map(1:nblocks) do col
  feop_row_col = feop[1,col]
  snaps_col = sols[col]
  norm_style_col = norm_style[col]
  norm_matrix = get_norm_matrix(feop,norm_style_col)
  basis_space_nnz,basis_time = compress(rbinfo,feop_row_col,snaps_col,norm_matrix,μ)
  basis_space = recast(basis_space_nnz)
  basis_space,basis_time,norm_matrix
end
bases_space = getindex.(blocks,1)
bases_time = getindex.(blocks,2)
norm_matrix = getindex.(blocks,3)

bs_primal,bs_dual = bases_space
nm_primal, = norm_matrix
feop_row_col = feop[1,2]
supr_col = space_supremizers(bs_dual,feop_row_col,μ)
B = -assemble_matrix((p,v) -> ∫(p*(∇⋅(v)))dΩ,trial_p,test_u)
@assert supr_col == B*bs_dual
copy_supr_col = copy(supr_col)
ℓ∞(copy_supr_col'*copy_supr_col - Float.(I(size(supr_col,2)))) - eps()
ℓ∞(copy_supr_col'*bs_primal) - eps()
gram_schmidt!(copy_supr_col,bs_primal,nm_primal)
bs_primal_supr = hcat(bs_primal,copy_supr_col)

_,σ,_ = svd(bs_primal_supr'*B*bs_dual)
norm(σ)

_bases_space = add_space_supremizers(bases_space,feop,norm_matrix,μ)
@assert _bases_space[1] == bs_primal_supr
@assert _bases_space[2] == bs_dual

nparams,time_ndofs=length(μ),get_time_ndofs(fesolver)
for nb in 1:nblocks
  nzm = NnzArray(sols[nb])
  full_val = recast(nzm)
  bs = rbspace[nb].basis_space
  bt = rbspace[nb].basis_time
  println(norm(full_val - bs*bs'*full_val)/norm(full_val))
  m2 = change_mode(nzm)
  println(norm(m2 - bt*bt'*m2)/norm(m2))
end

rbspace[1].basis_space'*rbspace[1].basis_space
rbspace[2].basis_space'*rbspace[2].basis_space
rbspace[1].basis_time'*rbspace[1].basis_time
rbspace[2].basis_time'*rbspace[2].basis_time
