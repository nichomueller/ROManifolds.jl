op,solver = feop,fesolver
K = 2
μ = realization(op,K)
t = solver.dt

V = get_test(op)#[1]
v = get_fe_basis(V)
U = get_trial(op)(nothing,nothing) #[1]
du = get_trial_fe_basis(U)

ode_op = get_algebraic_operator(op)
ode_cache = allocate_cache(ode_op,μ)
update_cache!(ode_cache,ode_op,μ,t)
nfree = num_free_dofs(test)
u = PTArray([zeros(nfree) for _ = 1:K])
vθ = similar(u)
vθ .= 1.0
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(op)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)

dca = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ) # dca = ∫ₚ(aμt(μ,t)*∇(v)⊙∇(du),dΩ)
dch = ∫ₚ(hμt(μ,t)*v,dΓn)
dcm = ∫ₚ(v*∂ₚt(xh),dΩ) # dcm = ∫ₚ(v⋅∂ₚt(xh[1]),dΩ)
quad = dΩ.quad
x = get_cell_points(quad)
b = change_domain(v*∂ₚt(xh),quad.trian,quad.data_domain_style)
bx = b(x)

# Gridap
g_ok(x,t) = g(x,μ[1],t)
g_ok(t) = x->g_ok(x,t)
trial_ok = TransientTrialFESpace(V,g_ok)
du_ok = get_trial_fe_basis(trial_ok(t))
m_ok(t,dut,v) = ∫(v*dut)dΩ
lhs_ok(t,du,v) = ∫(a(μ[1],t)*∇(v)⋅∇(du))dΩ
rhs_ok(t,v) = ∫(f(μ[1],t)*v)dΩ + ∫(h(μ[1],t)*v)dΓn
feop_ok = TransientAffineFEOperator(m_ok,lhs_ok,rhs_ok,trial_ok,V)
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

dc_ok = ∫(a(μ[1],t)*∇(v)⋅∇(du_ok))dΩ # dca_ok = ∫(a(μ[1],t)*∇(v)⊙∇(du_ok))dΩ
dch_ok = ∫(h(μ[1],t)*v)dΓn
dcm_ok = ∫(v*∂t(xh_ok))dΩ # dcm_ok = ∫(v⋅∂t(xh_ok))dΩ

b_ok = change_domain(v*∂t(xh_ok),quad.trian,quad.data_domain_style)
bx_ok = b_ok(x)

# TESTS

function runtest_matrix(dc,dc_ok,meas)
  strian = get_triangulation(meas)
  scell_mat = get_contribution(dc,meas)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(U(μ,t),cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(U(μ,t),trian)

  scell_mat_ok = get_contribution(dc_ok,strian)
  cell_mat_ok,trian_ok = move_contributions(scell_mat_ok,strian)
  @assert ndims(eltype(cell_mat_ok)) == 2
  cell_mat_c_ok = attach_constraints_cols(trial_ok(t),cell_mat_ok,trian_ok)
  cell_mat_rc_ok = attach_constraints_rows(test,cell_mat_c_ok,trian_ok)
  rows_ok = get_cell_dof_ids(test,trian_ok)
  cols_ok = get_cell_dof_ids(trial_ok(t),trian_ok)

  @check all(rows .== rows_ok)
  @check all(cols .== cols_ok)
  test_ptarray(cell_mat,cell_mat_ok)
  test_ptarray(cell_mat_rc,cell_mat_rc_ok)
end

function runtest_vector(dc,dc_ok,meas)
  strian = get_triangulation(meas)
  scell_vec = get_contribution(dc,meas)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)

  scell_vec_ok = get_contribution(dc_ok,strian)
  cell_vec_ok,trian_ok = move_contributions(scell_vec_ok,strian)
  @assert ndims(eltype(cell_vec_ok)) == 1
  cell_vec_r_ok = attach_constraints_rows(test,cell_vec_ok,trian_ok)
  rows_ok = get_cell_dof_ids(test,trian_ok)

  @check all(rows .== rows_ok)
  test_ptarray(cell_vec,cell_vec_ok)
  test_ptarray(cell_vec_r,cell_vec_r_ok)
end

runtest_matrix(dca,dc_ok,dΩ)
runtest_vector(dch,dch_ok,dΓn)
runtest_vector(dcm,dcm_ok,dΩ)

dj1 = op.jacs[1](μ,t,xh,du,v)
dj1_by_1 = 1. * dj1
test_ptarray(dj1[Ω],dj1_by_1[Ω])

dc1 = ∫ₚ(v*∂ₚt(xh) + aμt(μ,t)*∇(v)⋅∇(xh) - fμt(μ,t)*v,dΩ)
dc2 = ∫ₚ(hμt(μ,t)*v,dΓn)
dc3 = dc1-dc2
dc1_ok = ∫(v*∂t(xh_ok))dΩ + ∫(a(μ[1],t)*∇(v)⋅∇(xh_ok))dΩ - ∫(f(μ[1],t)*v)dΩ
dc2_ok = ∫(h(μ[1],t)*v)dΓn
dc3_ok = dc1_ok-dc2_ok
test_ptarray(dc3[Ω],dc3_ok[Ω])
test_ptarray(dc3[Γn],dc3_ok[Γn])
