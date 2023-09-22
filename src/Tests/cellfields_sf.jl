K = 2
μ = realization(feop,K)
t = solver.dt
strian = Ω

V = get_test(feop)
v = get_fe_basis(test)
U = get_trial(feop)(nothing,nothing)
du = get_trial_fe_basis(U)

x = get_cell_points(dΩ.quad)
cf = aμt(μ,t)*∇(v)⋅∇(du)
result = cf(x)

# Gridap
cf_ok = a(μ[1],dt)*∇(v)⋅∇(du)
result_ok = cf_ok(x)

test_ptarray(result,result_ok)

# test affinity
affine_cf = fμt(μ,dt)*v
affine_pt = affine_cf(x)
@assert isa(affine_pt,AffinePTArray)
