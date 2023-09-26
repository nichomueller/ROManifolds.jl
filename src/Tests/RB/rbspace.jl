params = realization(feop,10)

# collect_solutions(fesolver,feop,params)
uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
ode_op = get_algebraic_operator(feop)
uu0 = get_free_dof_values(uh0(params))
uμst = PODESolution(fesolver,ode_op,params,uu0,t0,tf)
num_iter = Int(tf/fesolver.dt)
snaps = allocate_solution(ode_op,num_iter)
for (u,t,n) in uμst
  printstyled("Computing fe solution at time $t for every parameter\n";color=:blue)
  snaps[n] = get_solution(ode_op,u)
end
snaps = Snapshots(snaps)

# get_reduced_basis(info,feop,snaps)
nzm = NnzArray(snaps)
basis_space = tpod(nzm,nothing;ϵ=1e-4)
compressed_nza = prod(basis_space,nzm)
compressed_nza_t = change_mode(compressed_nza)
basis_time = tpod(compressed_nza_t;ϵ=1e-4)
rbspace = RBSpace(basis_space,basis_time)

# change mode
# space_ndofs = num_space_dofs(nzm)
# time_ndofs = num_time_dofs(nzm)
# nparams = num_params(nzm)
# idx = time_param_idx(time_ndofs,nparams)

# mode2 = zeros(time_ndofs,space_ndofs*nparams)
# @inbounds for (i,col) = enumerate(eachcol(idx))
#   mode2[i,:] = reshape(nzm.nonzero_val[:,col]',:)
# end
