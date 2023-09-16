op,solver = feop,fesolver
μ = realization(op,2)
t = solver.dt
strian = Ω

vec_cache = PTArray([zeros(test.nfree) for _ = 1:2])
V = get_test(op)
v = get_fe_basis(V)
U = PTrialFESpace(vec_cache,V)
du = get_trial_fe_basis(U)

x = get_cell_points(dΩ.quad)
q = aμt(μ,t)*∇(v)⋅∇(du)
resq = q(x)
res1 = resq.array[1]

# Gridap
gok(x,t) = μ[1][1]*exp(-x[1]/μ[1][2])*abs(sin(t/μ[1][3]))
gok(t) = x->gok(x,t)
trial_ok = TransientTrialFESpace(test,gok)
feop_ok = TransientAffineFEOperator(res,jac,jac_t,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
uF_ok = similar(u[1])
uF_ok .= 1.0
ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
Us_ok,_,_ = ode_cache_ok
uh_ok = EvaluationFunction(Us_ok[1],uF_ok)
dxh_ok = ()
for i in 1:get_order(op)
  dxh_ok = (dxh_ok...,uh_ok)
end
xh_ok = TransientCellField(uh_ok,dxh_ok)
qok = a(μ[1],dt)*∇(v)⋅∇(xh_ok)
res1_ok = qok(x)


typeof(res1) == typeof(res1_ok) # true
all(res1 .== res1_ok) # true

U = trial(μ,t)
cell_vals = scatter_free_and_dirichlet_values(U,vec_cache,U.dirichlet_values)
cell_basis = get_data(v)
# lazy_map(linear_combination,cell_vals,cell_basis)
lazy_arrays = map(eachindex(cell_vals)) do i
  ai = _getter_at_ind(i,cell_vals,cell_basis)
  lazy_map(linear_combination,ai...)
end
FEFunction(U,vec_cache)
