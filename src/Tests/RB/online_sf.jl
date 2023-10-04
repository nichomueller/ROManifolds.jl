rbres,rbjac = rbrhs,rblhs
sols_test,params_test = load_test(info,feop,fesolver)
res_cache,jac_cache = allocate_sys_cache(feop,fesolver,rbspace,sols_test,params_test)
  # times = get_times(fesolver)
  # ode_op = get_algebraic_operator(feop)
  # ode_cache = allocate_cache(ode_op,params,times)
  # b = allocate_residual(ode_op,sols_test,ode_cache)
  # A = allocate_jacobian(ode_op,sols_test,ode_cache)

  # rb_ndofs = num_rb_dofs(rbspace)
  # ncoeff = length(params)
  # coeff = zeros(Float,rb_ndofs,rb_ndofs)
  # ptcoeff = PTArray([zeros(Float,rb_ndofs,rb_ndofs) for _ = 1:ncoeff])

  # k = RBContributionMap()
  # rbres = testvalue(RBAffineDecomposition{Float},feop;vector=true)
  # rbjac = testvalue(RBAffineDecomposition{Float},feop;vector=false)
  # res_contrib_cache = return_cache(k,rbres.basis_space,rbres.basis_time)
  # jac_contrib_cache = return_cache(k,rbjac.basis_space,rbjac.basis_time)

st_mdeim = info.st_mdeim
trian = get_domains(rbres)
coeff_cache,rb_cache = res_cache
t = Γn
rbrest = rbres[t]

coeff = rhs_coefficient!(coeff_cache,feop,fesolver,rbrest,trian,sols_test,params_test;st_mdeim)
  times = get_times(fesolver)
  rcache,ccache... = coeff_cache
  ndofs = num_free_dofs(feop.test)
  setsize!(rcache,(ndofs,))
  red_idx = rbrest.integration_domain.idx
  red_times = rbrest.integration_domain.times
  red_trian = rbrest.integration_domain.trian
  strian = substitute_trian(red_trian,trian)
  meas = map(t->get_measure(feop,t),strian)
  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,params_test,red_times)

  if length(red_times) < length(times)
    b = get_array(rcache;len=length(red_times)*length(params_test))
    time_idx = findall(x->x in red_times,times)
    idx = param_time_idx(time_idx,length(params_test))
    _sols = PTArray(sols_test[idx])
  else
    b = get_array(rcache)
    _sols = sols_test
  end
  nzm = collect_residuals!(b,fesolver,ode_op,_sols,params_test,red_times,ode_cache,red_trian,meas...)
    # dt,θ = fesolver.dt,fesolver.θ
    # dtθ = θ == 0.0 ? dt : dt*θ
    # ode_cache = update_cache!(ode_cache,ode_op,params_test,red_times)
    # nlop = PThetaMethodNonlinearOperator(ode_op,params_test,red_times,dtθ,sols_test,ode_cache,sols_test)
    # xhF = (sols_test,sols_test-sols_test)
    # Xh, = ode_cache
    # dxh = ()
    # for i in 2:get_order(ode_op)+1
    #   dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
    # end
    # xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
    # V = get_test(feop)
    # v = get_fe_basis(V)
    # meas = map(t->get_measure(feop,t),strian)
    # vecdata = collect_cell_vector(V,feop.res(params_test,red_times,xh,v,meas),red_trian)
    # assemble_vector_add!(b,feop.assem,vecdata)
  q = nzm[red_idx,:]
  csolve,crecast = ccache
  time_ndofs = length(rbrest.integration_domain.times)
  nparams = Int(size(q,2)/time_ndofs)
  _q = change_order(q,time_ndofs)
    # idx = reorder_col_idx(time_ndofs,nparams)
    # mode2 = zeros(Float,size(q))
    # @inbounds for i = axes(idx,1)
    #   mode2[:,idx[i,:]] = q[:,idx[:,i]]
    # end
  _coeff = mdeim_solve!(csolve,rbrest.mdeim_interpolation,_q)
  # coeff = recast_coefficient!(crecast,_coeff)
    Qs = Int(size(_coeff,1))
    Nt = Int(size(_coeff,2)/nparams)
    setsize!(crecast,(Nt,Qs))
    ptarray = get_array(crecast)

    @inbounds for n = eachindex(ptarray)
      ptarray[n] .= _coeff[:,(n-1)*Nt+1:n*Nt]'
    end

contrib = rb_contribution!(rb_cache,rbrest,coeff)

trian = get_domains(rbjac[1])
coeff_cache,rb_cache = jac_cache
t = Ω
rbjact = rbjac[1][t]
coeff = lhs_coefficient!(coeff_cache,feop,fesolver,rbjact,trian,sols_test,params_test;st_mdeim,i=1)
  # jcache,scache... = coeff_cache
  # q = assemble_lhs!(jcache,feop,fesolver,rbjact,trian,sols_test,params_test;i=1)
  # csolve,crecast = scache
  # time_ndofs = length(rbjact.integration_domain.times)
  # nparams = Int(size(q,2)/time_ndofs)
  # _q = change_order(q,time_ndofs)
  # _coeff = mdeim_solve!(csolve,rbjact.mdeim_interpolation,reshape(_q,:,nparams))
  # recast_coefficient!(crecast,rbjact.basis_time,_coeff)
contrib = rb_contribution!(rb_cache,rbjact,coeff)

# HIGH LEVEL
_snaps_test,_params_test = load_test(info,feop,fesolver)
_x = initial_guess(sols,params,_params_test)
_rhs_cache,_lhs_cache = allocate_sys_cache(feop,fesolver,rbspace,_snaps_test,_params_test)
# rhs = collect_rhs_contributions!(rhs_cache,info,feop,fesolver,rbrhs,x,params_test)
_coeff_cache,_rb_cache = _rhs_cache
_trian = get_domains(rbres)
st_mdeim = info.st_mdeim

_rb_res_contribs = PTArray{Vector{Float}}[]
for t in _trian
  _rbrest = rbres[t]
  _coeff = rhs_coefficient!(_coeff_cache,feop,fesolver,_rbrest,_trian,_x,_params_test;st_mdeim)
  _contrib = rb_contribution!(_rb_cache,_rbrest,_coeff)
  push!(_rb_res_contribs,_contrib)
end
sum(_rb_res_contribs)
