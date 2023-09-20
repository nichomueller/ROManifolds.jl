op,solver = feop,fesolver
K = 2
μ = realization(op,K)
t = solver.dt
strian = Ω

V = get_test(op)#[1]
v = get_fe_basis(V)
U = get_trial(op)(nothing,nothing)#[1](nothing,nothing)
du = get_trial_fe_basis(U)

x = get_cell_points(dΩ.quad)
q = aμt(μ,t)*∇(v)⋅∇(du)
resq = q(x)
res1 = resq[1]

# Gridap
qok = a(μ[1],dt)*∇(v)⋅∇(du)
res1_ok = qok(x)

typeof(res1) == typeof(res1_ok) # true
all(res1 .== res1_ok) # true

# test affinity
fcell = fμt(μ,dt)*v
typeof(fcell.args[1].cell_field)
