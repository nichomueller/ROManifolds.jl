μ = realization(feop,10)
sols = collect_solutions(fesolver,feop,μ)
rbspace = reduced_basis(info,feop,sols,fesolver,μ)
  # nzm = NnzArray(sols)
  # basis_space = tpod(nzm,nothing;ϵ=1e-4)
  # compressed_nza = prod(basis_space,nzm)
  # compressed_nza_t = change_mode(compressed_nza)
  # basis_time = tpod(compressed_nza_t;ϵ=1e-4)
  # rbspace = RBSpace(basis_space,basis_time)

nparams,time_ndofs=length(μ),get_time_ndofs(fesolver)
nzm = NnzArray(sols)
full_val = recast(nzm)
bs = rbspace.basis_space
bt = rbspace.basis_time
maximum(abs.(full_val - bs*bs'*full_val)) <= ϵ*10
m2 = change_mode(nzm)
maximum(abs.(m2 - bt*bt'*m2)) <= ϵ*10
