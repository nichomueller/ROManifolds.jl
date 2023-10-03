feop,solver = feop,fesolver
K = 2
μ = realization(feop,K)
t = solver.dt

V = get_test(feop)
v = get_fe_basis(V)
U = get_trial(feop)(nothing,nothing)
du = get_trial_fe_basis(U)

ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,t)
update_cache!(ode_cache,ode_op,μ,t)
nfree = num_free_dofs(test)
u = PTArray([zeros(nfree) for _ = 1:K])
vθ = similar(u)
vθ .= 1.0
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(feop)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)

dca = ∫(aμt(μ,t)*∇(v)⋅∇(du))dΩ
dch = ∫(hμt(μ,t)*v)dΓn
dcm = ∫(v*∂ₚt(xh))dΩ
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
for i in 1:get_order(feop)
  dxh_ok = (dxh_ok...,uh_ok)
end
xh_ok = TransientCellField(uh_ok,dxh_ok)

dca_ok = ∫(a(μ[1],t)*∇(v)⋅∇(du_ok))dΩ
dch_ok = ∫(h(μ[1],t)*v)dΓn
dcm_ok = ∫(v*∂t(xh_ok))dΩ

b_ok = change_domain(v*∂t(xh_ok),quad.trian,quad.data_domain_style)
bx_ok = b_ok(x)

# TESTS

function runtest_matrix(dc,dca_ok,strian)
  scell_mat = get_contribution(dc,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(U(μ,t),cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(U(μ,t),trian)

  scell_mat_ok = get_contribution(dca_ok,strian)
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

function runtest_vector(dc,dca_ok,strian)
  scell_vec = get_contribution(dc,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)

  scell_vec_ok = get_contribution(dca_ok,strian)
  cell_vec_ok,trian_ok = move_contributions(scell_vec_ok,strian)
  @assert ndims(eltype(cell_vec_ok)) == 1
  cell_vec_r_ok = attach_constraints_rows(test,cell_vec_ok,trian_ok)
  rows_ok = get_cell_dof_ids(test,trian_ok)

  @check all(rows .== rows_ok)
  test_ptarray(cell_vec,cell_vec_ok)
  test_ptarray(cell_vec_r,cell_vec_r_ok)
end

runtest_matrix(dca,dca_ok,Ω)
runtest_vector(dch,dch_ok,Γn)
runtest_vector(dcm,dcm_ok,Ω)

dj1 = feop.jacs[1](μ,t,xh,du,v)
dj1_by_1 = 1. * dj1
test_ptarray(dj1[Ω],dj1_by_1[Ω])

dc1 = ∫(v*∂ₚt(xh))dΩ + ∫(aμt(μ,t)*∇(v)⋅∇(xh))dΩ - ∫(fμt(μ,t)*v)dΩ
dc2 = ∫(hμt(μ,t)*v)dΓn
dc3 = dc1-dc2
dc1_ok = ∫(v*∂t(xh_ok))dΩ + ∫(a(μ[1],t)*∇(v)⋅∇(xh_ok))dΩ - ∫(f(μ[1],t)*v)dΩ
dc2_ok = ∫(h(μ[1],t)*v)dΓn
dc3_ok = dc1_ok-dc2_ok
test_ptarray(dc3[Ω],dc3_ok[Ω])
test_ptarray(dc3[Γn],dc3_ok[Γn])

test_ptarray((dcm + dch_ok)[Ω],(dcm_ok + dch_ok)[Ω])
test_ptarray((dch_ok + dcm)[Ω],(dch_ok + dcm_ok)[Ω])
test_ptarray((dcm - dch_ok)[Ω],(dcm_ok - dch_ok)[Ω])
test_ptarray((dch_ok - dcm)[Ω],(dch_ok - dcm_ok)[Ω])

# One param, multiple times
feop,solver = feop,fesolver
μ = μ[1]
times = [solver.dt,2*solver.dt]

V = get_test(feop)
v = get_fe_basis(V)
U = get_trial(feop)(nothing,nothing)
du = get_trial_fe_basis(U)

ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,times)
update_cache!(ode_cache,ode_op,μ,times)
nfree = num_free_dofs(test)
u = PTArray([zeros(nfree) for _ = eachindex(times)])
vθ = similar(u)
vθ .= 1.0
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(feop)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)

dca = ∫(aμt(μ,times)*∇(v)⋅∇(du))dΩ
dch = ∫(hμt(μ,times)*v)dΓn
dcm = ∫(v*∂ₚt(xh))dΩ

# Gridap
g_ok(x,t) = g(x,μ,t)
g_ok(t) = x->g_ok(x,t)
trial_ok = TransientTrialFESpace(V,g_ok)
du_ok = get_trial_fe_basis(trial_ok(t))
m_ok(t,dut,v) = ∫(v*dut)dΩ
lhs_ok(t,du,v) = ∫(a(μ,t)*∇(v)⋅∇(du))dΩ
rhs_ok(t,v) = ∫(f(μ,t)*v)dΩ + ∫(h(μ,t)*v)dΓn
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

dca_ok = ∫(a(μ,t)*∇(v)⋅∇(du_ok))dΩ
dch_ok = ∫(h(μ,t)*v)dΓn
dcm_ok = ∫(v*∂t(xh_ok))dΩ

runtest_matrix(dca,dca_ok,Ω)
runtest_vector(dch,dch_ok,Γn)
runtest_vector(dcm,dcm_ok,Ω)
