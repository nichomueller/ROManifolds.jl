op,solver = feop,fesolver
ode_op = get_algebraic_operator(op)
ode_cache = allocate_cache(ode_op,μ)
μ = realization(op,2)
t = solver.dt
ode_cache = update_cache!(ode_cache,ode_op,μ,t)

vec_cache = PTArray([zeros(test.nfree) for _ = 1:2])
V = get_test(op)
v = get_fe_basis(V)
U = PTTrialFESpace(vec_cache,V)
du = get_trial_fe_basis(U)

int = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
dc = evaluate(int)
matdata = collect_cell_matrix(U(μ,t),V,dc)
A_ok = allocate_jacobian(ode_op,vec_cache,ode_cache)

# Gridap
gok(x,t) = μ[1][1]*exp(-x[1]/μ[1][2])*abs(sin(t/μ[1][3]))
gok(t) = x->gok(x,t)
resok(t,u,v) = ∫(v*∂t(u))dΩ + ∫(a(μ,t)*∇(v)⋅∇(u))dΩ - ∫(f(μ,t)*v)dΩ - ∫(h(μ,t)*v)dΓn
jacok(t,u,du,v) = ∫(∇(v)⋅∇(du))dΩ
jacok_t(t,u,dut,v) = ∫(v*dut)dΩ
trial_ok = TransientTrialFESpace(test,gok)
feop_ok = TransientFEOperator(resok,jacok,jacok_t,trial_ok,test)
du_ok = get_trial_fe_basis(trial_ok(t))
dcok = ∫(a(μ[1],t)*∇(v)⋅∇(du_ok))dΩ
matdata_ok = collect_cell_matrix(trial_ok(t),V,dcok)
uh_ok = zero(test)
A_ok = allocate_jacobian(feop_ok,uh_ok,nothing)
