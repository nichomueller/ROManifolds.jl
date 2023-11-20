sols,params = load(rbinfo,(Snapshots,Table))
rbspace = reduced_basis(rbinfo,feop,sols;nsnaps_state)
  # nzm = NnzArray(sols)
  # basis_space = tpod(nzm,nothing;ϵ=1e-4)
  # compressed_nza = prod(basis_space,nzm)
  # compressed_nza_t = change_mode(compressed_nza)
  # basis_time = tpod(compressed_nza_t;ϵ=1e-4)
  # rbspace = RBSpace(basis_space,basis_time)

nparams,time_ndofs=length(params),get_time_ndofs(fesolver)
nzm = NnzMatrix(sols[1:nsnaps_state];nparams=nsnaps_state)
full_val = recast(nzm)
bs = rbspace.basis_space
bt = rbspace.basis_time
maximum(abs.(full_val - bs*bs'*full_val)) <= ϵ*10
m2 = change_mode(nzm)
maximum(abs.(m2 - bt*bt'*m2)) <= ϵ*10
