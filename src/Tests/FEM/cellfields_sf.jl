feop,solver = feop,fesolver
K = 2
μ = realization(feop,K)
Nt = 3
times = [dt,2dt,3dt]
N = K*Nt
strian = Ω

V = get_test(feop)
dv = get_fe_basis(test)
U = get_trial(feop)(nothing,nothing)
du = get_trial_fe_basis(U)

c = 1.0
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,times)
update_cache!(ode_cache,ode_op,μ,times)
nfree = num_free_dofs(test)
u = PTArray([zeros(nfree) for _ = 1:N])
vθ = similar(u)
vθ .= c
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(feop)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)

x = get_cell_points(dΩ.quad)

# THIS WORKS
ca = (aμt(μ,times)*∇(dv)⋅∇(du))(x)
ch = (hμt(μ,times)*dv)(x)
for n = 1:N
  p = μ[slow_idx(n,Nt)]
  t = times[fast_idx(n,Nt)]
  ca_ok = (a(p,t)*∇(dv)⋅∇(du))(x)
  ch_ok = (h(p,t)*dv)(x)
  test_ptarray(ca,ca_ok;n)
  test_ptarray(ch,ch_ok;n)
end

cr = (aμt(μ,times)*∇(dv)⋅∇(xh))(x)
function get_cell_field_gridap(n)
  p,t = μ[slow_idx(n,Nt)],times[fast_idx(n,Nt)]
  g_ok(x,t) = g(x,p,t)
  g_ok(t) = x->g_ok(x,t)
  trial_ok = TransientTrialFESpace(V,g_ok)
  m_ok(t,dut,dv) = ∫(dv*dut)dΩ
  lhs_ok(t,du,dv) = ∫(a(p,t)*∇(dv)⋅∇(du))dΩ
  rhs_ok(t,dv) = ∫(f(p,t)*dv)dΩ + ∫(h(p,t)*dv)dΓn
  feop_ok = TransientAffineFEOperator(m_ok,lhs_ok,rhs_ok,trial_ok,V)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  uF_ok = similar(u[1])
  uF_ok .= c
  ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Us_ok,_,_ = ode_cache_ok
  uh_ok = EvaluationFunction(Us_ok[1],uF_ok)
  dxh_ok = ()
  for i in 1:get_order(feop)
    dxh_ok = (dxh_ok...,uh_ok)
  end
  xh_ok = TransientCellField(uh_ok,dxh_ok)
  xh_ok,(a(p,t)*∇(dv)⋅∇(xh_ok))(x)
end

for n = 1:N
  xh_ok,_ = get_cell_field_gridap(n)
  #test_ptarray(xh.cellfield.cell_dof_values,xh_ok.cellfield.cell_dof_values;n)
  test_ptarray(xh.cellfield.dirichlet_values,xh_ok.cellfield.dirichlet_values;n)
end

for n = 1:N
  p = μ[slow_idx(n,Nt)]
  t = times[fast_idx(n,Nt)]
  _,cr_ok = get_cell_field_gridap(n)
  test_ptarray(cr,cr_ok;n)
end

# test affinity
affine_cf = fμt(μ,dt)*dv
affine_pt = affine_cf(x)
@assert isa(affine_pt,AffinePTArray)


μ = rand(3)
t = [0,1]
cf = aμt(μ,t)*∇(dv)⋅∇(du)
result = cf(x)
cf_ok = a(μ,t[1])*∇(dv)⋅∇(du)
result_ok = cf_ok(x)
test_ptarray(result,result_ok)
