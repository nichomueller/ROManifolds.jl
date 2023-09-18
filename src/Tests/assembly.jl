op,solver = feop,fesolver
μ = realization(op,2)
t = solver.dt
u = PTArray([zeros(test.nfree) for _ = 1:2])
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
p = μ[1]
gok(x,t) = g(x,p,t)
gok(t) = x->gok(x,t)
resok(t,u,v) = ∫(v*∂t(u))dΩ + ∫(a(p,t)*∇(v)⋅∇(u))dΩ - ∫(f(p,t)*v)dΩ - ∫(h(p,t)*v)dΓn
jacok(t,u,du,v) = ∫(a(p,t)*∇(v)⋅∇(du))dΩ
jacok_t(t,u,dut,v) = ∫(v*dut)dΩ
trial_ok = TransientTrialFESpace(test,gok)
feop_ok = TransientFEOperator(resok,jacok,jacok_t,trial_ok,test)
du_ok = get_trial_fe_basis(trial_ok(t))
uh_ok = zero(test)

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

b_ok = allocate_residual(feop_ok,uh_ok,nothing)
vecdata_ok = collect_cell_vector(test,feop_ok.res(t,xh_ok,v))
assemble_vector_add!(b_ok,feop_ok.assem_t,vecdata_ok)
test_ptarray(b_ok,b)
