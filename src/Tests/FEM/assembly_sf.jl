op,solver = feop,fesolver
K = 2
μ = realization(op,K)
t = solver.dt
nfree = num_free_dofs(test)
u = PTArray([zeros(nfree) for _ = 1:K])
vθ = similar(u)
vθ .= 1.0
ode_op = get_algebraic_operator(op)
ode_cache = allocate_cache(ode_op,μ)
ode_cache = update_cache!(ode_cache,ode_op,μ,t)
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(op)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)

A = allocate_jacobian(op,uh,ode_cache)
_matdata_jacobians = fill_jacobians(op,μ,t,xh,(1.,1/t))
matdata = _vcat_matdata(_matdata_jacobians)
assemble_matrix_add!(A,op.assem,matdata)

v = get_fe_basis(test)
b = allocate_residual(op,uh,ode_cache)
vecdata = collect_cell_vector(test,op.res(μ,t,xh,v))
assemble_vector_add!(b,op.assem,vecdata)

# Gridap
test_ok = test
pp = μ[1]
g_ok(x,t) = g(x,pp,t)
g_ok(t) = x->g_ok(x,t)
a_ok(t,u,v) = ∫(a(μ[1],t)*∇(v)⋅∇(u))dΩ
b_ok(t,v) = ∫(v*f(μ[1],t))dΩ + ∫(v*h(μ[1],t))dΓn
m_ok(t,ut,v) = ∫(ut*v)dΩ
trial_ok = TransientTrialFESpace(test_ok,g_ok)
feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test_ok)

du_ok = get_trial_fe_basis(trial_ok(t))

ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
uF_ok = similar(u[1])
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

A_ok = allocate_jacobian(feop_ok,uh_ok,nothing)
_matdata_jacobians_ok = fill_jacobians(feop_ok,t,xh_ok,(1.,1/t))
matdata_ok = Gridap.ODEs.TransientFETools._vcat_matdata(_matdata_jacobians_ok)
assemble_matrix_add!(A_ok,feop_ok.assem_t,matdata_ok)
test_ptarray(matdata_ok[1][1],matdata[1][1])
test_ptarray(A_ok,A)

bok = allocate_residual(feop_ok,uh_ok,nothing)
vecdata_ok = collect_cell_vector(test,feop_ok.res(t,xh_ok,v))
assemble_vector_add!(bok,feop_ok.assem_t,vecdata_ok)
test_ptarray(bok,b)
