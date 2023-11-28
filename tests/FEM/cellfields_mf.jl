K = 2
μ = realization(feop,K)
strian = Ω

v = get_fe_basis(test_u)
du = get_trial_fe_basis(trial_u(nothing,nothing))

x = get_cell_points(dΩ.quad)
cf = aμt(μ,dt)*∇(v)⊙∇(du)
result = cf(x)

# Gridap
cf_ok = a(μ[1],dt)*∇(v)⊙∇(du)
result_ok = cf_ok(x)

test_ptarray(result,result_ok)

# test affinity
affine_cf = fμt(μ,dt)⋅v
affine_pt = affine_cf(x)
@assert isa(affine_pt,AffinePTArray)

# nonlinearity
ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ)
update_cache!(ode_cache,ode_op,μ,dt)
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

xh1 = xh[1]
conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
ccf = v⊙(conv∘(xh1,∇(xh1)))
dccf = v⊙(dconv∘(du,∇(du),xh1,∇(xh1)))
