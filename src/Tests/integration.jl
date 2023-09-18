op,solver = feop,fesolver
μ = realization(op,2)
t = solver.dt
strian = Ω

vec_cache = PTArray([zeros(test.nfree) for _ = 1:2])
V = get_test(op)
v = get_fe_basis(V)
U = PTrialFESpace(vec_cache,V)
du = get_trial_fe_basis(U)

int = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
dc = evaluate(int)

hdc = evaluate(∫ₚ(hμt(μ,t)*v,dΓn))

ode_op = get_algebraic_operator(op)
ode_cache = allocate_cache(ode_op,μ)
update_cache!(ode_cache,ode_op,μ,t)
u = PTArray([zeros(test.nfree) for _ = 1:2])
vθ = similar(u)
vθ .= 1.0
Us,_,fecache = ode_cache
uh = EvaluationFunction(Us[1],vθ)
dxh = ()
for i in 1:get_order(op)
  dxh = (dxh...,uh)
end
xh = TransientCellField(uh,dxh)
mdc = evaluate(∫ₚ(v*∂ₚt(xh),dΩ))

# Gridap
g_ok(x,t) = g(x,μ[1],t)
g_ok(t) = x->g_ok(x,t)
trial_ok = TransientTrialFESpace(test,g_ok)
du_ok = get_trial_fe_basis(trial_ok(t))
dc_ok = ∫(a(μ[1],t)*∇(v)⋅∇(du_ok))dΩ

hdc_ok = ∫(h(μ[1],t)*v)dΓn

feop_ok = TransientAffineFEOperator(res,jac,jac_t,trial_ok,test)
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
mdc_ok = ∫(v*∂t(xh_ok))dΩ

# TESTS

function runtest_matrix(dc,dc_ok,strian)
  scell_mat = get_contribution(dc,strian)
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

function runtest_vector(dc,dc_ok,strian)
  scell_vec = get_contribution(dc,strian)
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

runtest_matrix(dc,dc_ok,strian)
runtest_vector(hdc,hdc_ok,Γn)
runtest_vector(mdc,mdc_ok,strian)

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
