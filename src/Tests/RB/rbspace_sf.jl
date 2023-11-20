sols,params = load(rbinfo,(Snapshots{Vector{Float}},Table))
rbspace = reduced_basis(rbinfo,feop,sols)
# nzm = NnzMatrix(sols[1:nsnaps_state];nparams=nsnaps_state)
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
norm(full_val - bs*bs'*full_val) / norm(full_val) <= ϵ
m2 = change_mode(nzm)
norm(m2 - bt*bt'*m2,Inf) / norm(m2) <= ϵ
