feop,solver = feop,fesolver
K = 2
μ = realization(feop,K)
Nt = 3
times = [dt,2dt,3dt]
N = K*Nt

V = get_test(feop)
dv = get_fe_basis(V)
U = get_trial(feop)(nothing,nothing)
du = get_trial_fe_basis(U)

ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,times)
update_cache!(ode_cache,ode_op,μ,times)
nfree = num_free_dofs(test)
u = PTArray([zeros(nfree) for _ = 1:N])
vθ = similar(u)
vθ .= 1.0
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(feop)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)

dca = ∫(aμt(μ,times)*∇(dv)⋅∇(xh))dΩ
dch = ∫(hμt(μ,times)*dv)dΓn
dcm = ∫(dv*∂ₚt(xh))dΩ
dc = dcm + dca - dch
dca1 = integrate(∫ₚ(aμt(μ,times)*∇(dv)⋅∇(xh),dΩ))
dch1 = integrate(∫ₚ(hμt(μ,times)*dv,dΓn))
dcm1 = integrate(∫ₚ(dv*∂ₚt(xh),dΩ))
dc1 = dcm1 + dca1 - dch1
test_ptarray(dc[Ω],dc1[Ω])

dcres = integrate(feop.res(μ,times,xh,dv))
dcjac1 = integrate(feop.jacs[1](μ,times,xh,du,dv))
dcjac2 = integrate(feop.jacs[2](μ,times,xh,du,dv))

# Gridap
function get_quantities_gridap(n)
  p,t = μ[slow_idx(n,Nt)],times[fast_idx(n,Nt)]
  g_ok(x,t) = g(x,p,t)
  g_ok(t) = x->g_ok(x,t)
  trial_ok = TransientTrialFESpace(V,g_ok)
  du_ok = get_trial_fe_basis(trial_ok(t))
  m_ok(t,dut,dv) = ∫(dv*dut)dΩ
  lhs_ok(t,du,dv) = ∫(a(p,t)*∇(dv)⋅∇(du))dΩ
  rhs_ok(t,dv) = ∫(f(p,t)*dv)dΩ + ∫(h(p,t)*dv)dΓn
  feop_ok = TransientAffineFEOperator(m_ok,lhs_ok,rhs_ok,trial_ok,V)
  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  uF_ok = similar(u[1])
  uF_ok .= 1.0
  ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Us_ok,_,_ = ode_cache_ok
  uh_ok = EvaluationFunction(Us_ok[1],uF_ok)
  dxh_ok = ()
  for i in 1:get_order(feop)
    dxh_ok = (dxh_ok...,uh_ok)
  end
  xh_ok = TransientCellField(uh_ok,dxh_ok)

  dca_ok = ∫(a(p,t)*∇(dv)⋅∇(xh_ok))dΩ
  dch_ok = ∫(h(p,t)*dv)dΓn
  dcm_ok = ∫(dv*∂t(xh_ok))dΩ

  dc_ok = dcm_ok + dca_ok - dch_ok
  dcres_ok = feop_ok.res(t,xh_ok,dv)
  dcjac1_ok = feop_ok.jacs[1](t,xh_ok,du,dv)
  dcjac2_ok = feop_ok.jacs[2](t,xh_ok,du,dv)

  return dc_ok,dcres_ok,dcjac1_ok,dcjac2_ok
end

for n = 1:N
  dc_ok,dcres_ok,dcjac1_ok,_ = get_quantities_gridap(n)
  test_ptarray(dc[Ω],dc_ok[Ω];n)
  test_ptarray(dcres[Ω],dcres_ok[Ω];n)
  test_ptarray(dcres[Γn],dcres_ok[Γn];n)
  test_ptarray(dcjac1[Ω],dcjac1_ok[Ω];n)
  # test_ptarray(dcjac2[Ω],dcjac2_ok[Ω];n)
end

dc_ok = integrate(feop.res(μ,times,xh,v))
dc = feop.res(μ,times,xh,v)[dΩ] + feop.res(μ,times,xh,v)[dΓn]
for t in get_domains(dc_ok)
  test_ptarray(dc_ok[t],dc[t])
end

ciao(μ,t,u,v) = ∫(v*∂ₚt(u))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ - ∫(fμt(μ,t)*v)dΩ - ∫(hμt(μ,t)*v)dΓn

dcc = ciao(μ,times,xh,v)
