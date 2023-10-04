# LOW LEVEL
nsnaps = info.nsnaps_system
snapsθ = recenter(fesolver,sols,params)
for i = eachindex(snapsθ)
  test_ptarray(snapsθ.snaps[i],sols.snaps[i])
end
_snapsθ,_μ = snapsθ[1:nsnaps],params[1:nsnaps]
times = get_times(fesolver)
ress,trian = collect_residuals_for_trian(fesolver,feop,_snapsθ,_μ,times)
  # ode_op = get_algebraic_operator(feop)
  # ode_cache = allocate_cache(ode_op,params,times)
  # b = allocate_residual(ode_op,_snapsθ,ode_cache)
  # dt,θ = fesolver.dt,fesolver.θ
  # dtθ = θ == 0.0 ? dt : dt*θ
  # ode_cache = update_cache!(ode_cache,ode_op,params,times)
  # nlop = PThetaMethodNonlinearOperator(ode_op,params,times,dtθ,_snapsθ,ode_cache,_snapsθ)
  # xhF = (_snapsθ,_snapsθ-_snapsθ)
  # Xh, = ode_cache
  # dxh = ()
  # for i in 2:get_order(ode_op)+1
  #   dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  # end
  # xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  # V = get_test(feop)
  # v = get_fe_basis(V)
  # dc = feop.res(params,times,xh,v)
  # trian = get_domains(dc)
  # bvec = Vector{typeof(b)}(undef,num_domains(dc))
  # for (n,t) in enumerate(trian)
  #   vecdata = collect_cell_vector(V,dc,t)
  #   assemble_vector_add!(b,feop.assem,vecdata)
  #   bvec[n] = copy(b)
  # end
  # nzm = NnzMatrix(bvec[1];nparams=length(params))
nzm,_trian = ress[2],Γn
full_val = recast(nzm)
basis_space,basis_time = compress(nzm;ϵ=info.ϵ)
proj_bs,proj_bt = compress_space_time(basis_space,basis_time,rbspace)
interp_idx_space,interp_idx_time = get_interpolation_idx(basis_space),get_interpolation_idx(basis_time)
entire_interp_idx_space = recast_idx(basis_space,interp_idx_space)

maximum(abs.(full_val - basis_space*basis_space'*full_val)) <= ϵ*10
m2 = change_mode(nzm)
maximum(abs.(m2 - basis_time*basis_time'*m2)) <= ϵ*10

interp_bs = basis_space[interp_idx_space,:]
lu_interp = if info.st_mdeim
  interp_bt = basis_time[interp_idx_time,:]
  interp_bst = LinearAlgebra.kron(interp_bt,interp_bs)
  lu(interp_bst)
else
  lu(interp_bs)
end

cell_dof_ids = get_cell_dof_ids(feop.test,_trian)
red_integr_cells = find_cells(entire_interp_idx_space,cell_dof_ids)
red_trian = view(_trian,red_integr_cells)
red_times = st_mdeim ? times[interp_idx_time] : times

i = 1
# ode_op = get_algebraic_operator(feop)
# ode_cache = allocate_cache(ode_op,params,times)
# A = allocate_jacobian(ode_op,_snapsθ,ode_cache)
# dt,θ = fesolver.dt,fesolver.θ
# dtθ = θ == 0.0 ? dt : dt*θ
# ode_cache = update_cache!(ode_cache,ode_op,params,times)
# nlop = PThetaMethodNonlinearOperator(ode_op,params,times,dtθ,_snapsθ,ode_cache,_snapsθ)
# @which collect_jacobians!(Val(true),A,nlop,_snapsθ;i)
# jacs_i,trian = jacobian!(A,nlop,_snapsθ,i,Val(true))
# @which NnzMatrix(map(NnzVector,jacs_i[1]);nparams=length(nlop.μ))
# nzm = NnzMatrix.(jacs_i;nparams=length(nlop.μ))
times = get_times(fesolver)
combine_projections = (x,y) -> i == 1 ? θ*x+(1-θ)*y : x-y
jacs,trian = collect_jacobians_for_trian(fesolver,feop,_snapsθ,_μ,times;i)
# compress_component(info,feop,jacs,trian,times,rbspace,rbspace;combine_projections)
  basis_space,basis_time = compress(jacs[1];ϵ=info.ϵ)
  proj_bs,proj_bt = compress_space_time(basis_space,basis_time,rbspace,rbspace;combine_projections)

  nzm = jacs[1]
  full_val = nzm.nonzero_val
  maximum(abs.(full_val - basis_space*basis_space'*full_val)) <= ϵ*10
  m2 = change_mode(nzm)
  maximum(abs.(m2 - basis_time*basis_time'*m2)) <= ϵ*10
