N = 10
u = sols[N]
p = params[N]

rbres,rbjac = rbrhs,rblhs
res_cache,jac_cache = allocate_sys_cache(feop,fesolver,rbspace,u,Table([p]))
st_mdeim = info.st_mdeim
trian = get_domains(rbres)
coeff_cache,rb_cache = res_cache
_trian = Ω
rbrest = rbres[_trian]
# coeff = rhs_coefficient!(coeff_cache,feop,fesolver,rbrest,trian,u,Table([p]);st_mdeim)
  rcache,scache... = coeff_cache
  red_integr_res = assemble_rhs!(rcache,feop,fesolver,rbrest,trian,u,Table([p]))

# CHECK THAT THE RESIDUALS/JACOBIANS ARE == TO GRIDAP'S RESIDUALS/JACOBIANS
# nlop = PThetaMethodNonlinearOperator(ode_op,p,times,dt*θ,u,ode_cache,u)
# my_res = residual(nlop,u,_trian,dΩ,dΓn)
times = get_times(fesolver)
dv = get_fe_basis(test)
du = get_trial_fe_basis(get_trial(feop)(nothing,nothing))
uF = copy(u)
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,p,times)
my_res = allocate_residual(ode_op,p,times,uF,ode_cache)
my_jac = allocate_jacobian(ode_op,p,times,uF,ode_cache)
ode_cache = update_cache!(ode_cache,ode_op,p,times)
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],uF)
dxh = ()
for i in 1:get_order(feop)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)
vecdata = collect_cell_vector(test,feop.res(p,times,xh,dv),_trian)
assemble_vector_add!(my_res,feop.assem,vecdata)
matdata = collect_cell_matrix(trial(p,times),test,feop.jacs[1](p,times,xh,du,dv),_trian)
assemble_matrix_add!(my_jac,feop.assem,matdata)

g_ok(x,t) = g(x,p,t)
g_ok(t) = x->g_ok(x,t)
trial_ok = TransientTrialFESpace(test,g_ok)
m_ok(t,dut,v) = ∫(v*dut)dΩ
lhs_ok(t,du,v) = ∫(a(p,t)*∇(v)⋅∇(du))dΩ
rhs_ok(t,v) = ∫(f(p,t)*v)dΩ + ∫(h(p,t)*v)dΓn
feop_ok = TransientAffineFEOperator(m_ok,lhs_ok,rhs_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)

gridap_res = []
gridap_jac = []
for (n,tn) in enumerate(times)
  uF_ok = copy(u[n])
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,tn)
  Us_ok,_,_ = ode_cache_ok
  uh_ok = EvaluationFunction(Us_ok[1],uF_ok)
  dxh_ok = ()
  for i in 1:get_order(feop)
    dxh_ok = (dxh_ok...,uh_ok)
  end
  xh_ok = TransientCellField(uh_ok,dxh_ok)
  vecdata_ok = collect_cell_vector(test,feop_ok.res(tn,xh_ok,dv),_trian)
  res_ok = assemble_vector(feop_ok.assem_t,vecdata_ok)
  push!(gridap_res,res_ok)
  println(isapprox(res_ok,my_res[n]))
  matdata_ok = collect_cell_matrix(feop_ok.trials[1](tn),test,feop_ok.jacs[1](tn,xh_ok,du,dv),_trian)
  jac_ok = assemble_matrix(feop_ok.assem_t,matdata_ok)
  push!(gridap_jac,jac_ok)
  println(isapprox(jac_ok,my_jac[n]))
end
