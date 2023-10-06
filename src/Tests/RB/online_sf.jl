rbres,rbjac = rbrhs,rblhs
sols_test,params_test = load_test(info,feop,fesolver)
res_cache,jac_cache = allocate_sys_cache(feop,fesolver,rbspace,sols_test,params_test)

trians = get_domains(rbres)
trian = Ω
rbrest = rbres[trian]
cache, = res_cache
test_affine_decomposition_rhs(cache,feop,fesolver,rbrest,trians,sols_test,params_test;st_mdeim)

rcache,scache... = cache

times = get_times(fesolver)
ndofs = num_free_dofs(feop.test)
setsize!(rcache,(ndofs,))

red_idx = rbrest.integration_domain.idx
red_times = rbrest.integration_domain.times
red_trian = rbrest.integration_domain.trian
strian = substitute_trian(red_trian,trians)
red_meas = map(t->get_measure(feop,t),strian)
full_meas = map(t->get_measure(feop,t),[trians...])

# test_red_meas(feop,fesolver,sols_test,params_test,red_meas...)
dv = get_fe_basis(feop.test)
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,params_test,times)
ode_cache = update_cache!(ode_cache,ode_op,params_test,times)
Xh, = ode_cache
dxh = ()
_xh = (sols_test,sols_test-sols_test)
for i in 2:get_order(feop)+1
  dxh = (dxh...,EvaluationFunction(Xh[i],_xh[i]))
end
xh = TransientCellField(EvaluationFunction(Xh[1],_xh[1]),dxh)
vecdata = collect_cell_vector(feop.test,feop.res(params_test,times,xh,dv),trian)
red_vecdata = collect_cell_vector(feop.test,feop.res(params_test,times,xh,dv,red_meas),trian)
err = maximum(map((x,y)->maximum(abs.(x-y)),(vecdata[1][1][n],red_vecdata[1][1][n])))

ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,params_test,red_times)

if length(red_times) < length(times)
  b = get_array(rcache;len=length(red_times)*length(params_test))
  time_idx = findall(x->x in red_times,times)
  _sols = map(x->getindex(x,time_idx),sols)
else
  b = get_array(rcache)
  _sols = sols_test
end
bfull = copy(b)
nzm = collect_residuals!(b,fesolver,ode_op,_sols,params_test,red_times,ode_cache,red_trian,red_meas...)
nzm_full = collect_residuals!(bfull,fesolver,ode_op,_sols,params_test,red_times,ode_cache,red_trian,full_meas...)
basis_space,_ = compress(nzm_full)
red_idx_full = sparse_to_full_idx(red_idx,nzm.nonzero_idx)
red_integr_res = nzm[red_idx_full,:]
err_res = maximum(abs.(nzm-nzm_full))
println("Resiudal difference for selected triangulation is $err_res")
