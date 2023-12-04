op,solver = feop,fesolver
K = 2
μ = realization(op,K)
times = [dt,2dt,3dt]
Nt = length(times)
N = K*Nt
nfree = num_free_dofs(test)
u = PTArray([zeros(nfree) for _ = 1:N])
vθ = similar(u)
vθ .= 1.0
ode_op = get_algebraic_operator(op)
ode_cache = allocate_cache(ode_op,μ,times)
ode_cache = update_cache!(ode_cache,ode_op,μ,times)
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(op)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)

A = allocate_jacobian(op,μ,times,uh,ode_cache)
_matdata_jacobians = fill_jacobians(op,μ,times,xh,(1.,1/dt))
matdata = _vcat_matdata(_matdata_jacobians)
assemble_matrix_add!(A,op.assem,matdata)

v = get_fe_basis(test)
b = allocate_residual(op,μ,times,uh,ode_cache)
vecdata = collect_cell_vector(test,op.res(μ,times,xh,v))
assemble_vector_add!(b,op.assem,vecdata)

# Gridap
function gridap_quantities_for_int(n::Int)
  p = μ[slow_idx(n,Nt)]
  t = times[fast_idx(n,Nt)]
  test_ok = test
  g_ok(x,t) = g(x,p,t)
  g_ok(t) = x->g_ok(x,t)
  a_ok(t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
  b_ok(t,v) = ∫(v*f(p,t))dΩ + ∫(v*h(p,t))dΓn
  m_ok(t,ut,v) = ∫(ut*v)dΩ
  trial_ok = TransientTrialFESpace(test_ok,g_ok)
  feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test_ok)

  ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
  uF_ok = similar(u[N])
  uF_ok .= 1.0
  ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
  Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,t)
  Us_ok,_,_ = ode_cache_ok
  uh_ok = EvaluationFunction(Us_ok[1],uF_ok)
  dxh_ok = ()
  for i in 1:get_order(op)
    dxh_ok = (dxh_ok...,uh_ok)
  end
  xh_ok = TransientCellField(uh_ok,dxh_ok)

  A_ok = allocate_jacobian(feop_ok,t,uh_ok,nothing)
  _matdata_jacobians_ok = fill_jacobians(feop_ok,t,xh_ok,(1.,1/dt))
  matdata_ok = Gridap.ODEs.TransientFETools._vcat_matdata(_matdata_jacobians_ok)
  assemble_matrix_add!(A_ok,feop_ok.assem_t,matdata_ok)

  bok = allocate_residual(feop_ok,t,uh_ok,nothing)
  vecdata_ok = collect_cell_vector(test,feop_ok.res(t,xh_ok,v))
  assemble_vector_add!(bok,feop_ok.assem_t,vecdata_ok)

  bok,A_ok,vecdata_ok,matdata_ok
end

for n = 1:N
  bok,Aok,vecdata_ok,matdata_ok = gridap_quantities_for_int(n)
  test_ptarray(matdata_ok[1][1],matdata[1][1];n)
  test_ptarray(Aok,A;n)
  test_ptarray(bok,b;n)
end
