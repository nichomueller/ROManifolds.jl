op,solver = feop,fesolver
μ = realization(op,2)
t = solver.dt

vec_cache = PTArray([zeros(test.nfree) for _ = 1:2])
V = get_test(op)
v = get_fe_basis(V)
U = PTTrialFESpace(vec_cache,V)
du = get_trial_fe_basis(U)

int = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
dc = evaluate(int)
strian = Ω
scell_mat = get_contribution(dc,strian)
cell_mat,trian = move_contributions(scell_mat,strian)
@assert ndims(eltype(testitem(cell_mat))) == 2
cell_mat_c = attach_constraints_cols(U(μ,t),cell_mat,trian)
cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
rows = get_cell_dof_ids(test,trian)
cols = get_cell_dof_ids(U(μ,t),trian)

# Gridap
gok(x,t) = μ[1][1]*exp(-x[1]/μ[1][2])*abs(sin(t/μ[1][3]))
gok(t) = x->gok(x,t)
trial_ok = TransientTrialFESpace(test,gok)
du_ok = get_trial_fe_basis(trial_ok(t))

dcok = ∫(a(μ[1],t)*∇(v)⋅∇(du_ok))dΩ
strian = Ω
scell_mat_ok = get_contribution(dcok,strian)
cell_mat_ok,trian_ok = move_contributions(scell_mat_ok,strian)
@assert ndims(eltype(cell_mat_ok)) == 2
cell_mat_c_ok = attach_constraints_cols(trial_ok(t),cell_mat_ok,trian_ok)
cell_mat_rc_ok = attach_constraints_rows(test,cell_mat_c_ok,trian_ok)
rows_ok = get_cell_dof_ids(test,trian_ok)
cols_ok = get_cell_dof_ids(trial_ok(t),trian_ok)

test_ptarray(cell_mat,cell_mat_ok)
test_ptarray(cell_mat_rc,cell_mat_rc_ok)
