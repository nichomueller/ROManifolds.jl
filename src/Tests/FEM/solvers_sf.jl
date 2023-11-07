K = 2
μ = realization(feop,K)
dtθ = dt*θ

# POISSON
ode_op = get_algebraic_operator(feop)
w = get_free_dof_values(uh0μ(μ))
sol = PODESolution(fesolver,ode_op,μ,w,t0,tf)

results = PTArray[]
for (uh,t) in sol
  push!(results,copy(uh))
end

n = 2
p,v = μ[n],w[n]
g_ok(x,t) = g(x,p,t)
g_ok(t) = x->g_ok(x,t)
a_ok(t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
b_ok(t,v) = ∫(v*f(p,t))dΩ + ∫(v*h(p,t))dΓn
m_ok(t,ut,v) = ∫(ut*v)dΩ

trial_ok = TransientTrialFESpace(test,g_ok)
feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_solver = ThetaMethod(LUSolver(),dt,θ)
sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,v,t0,tf)

results_ok = Vector{Float}[]
for (uh,t) in sol_gridap
  push!(results_ok,copy(uh))
end

for i in eachindex(results)
  test_ptarray(results[i],results_ok[i];n)
end

# affinity
a(x,μ,t) = 1
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = PTFunction(a,μ,t)

res(μ,t,u,v) = ∫ₚ(v*∂ₚt(u),dΩ) + ∫ₚ(aμt(μ,t)*∇(v)⊙∇(u),dΩ) - ∫ₚ(fμt(μ,t)*v,dΩ) - ∫ₚ(hμt(μ,t)*v,dΓn)
jac(μ,t,u,du,v) = ∫ₚ(aμt(μ,t)*∇(v)⋅∇(du),dΩ)
jac_t(μ,t,u,dut,v) = ∫ₚ(v*dut,dΩ)

feop = AffinePTFEOperator(res,jac,jac_t,pspace,trial,test)

ode_op = get_algebraic_operator(feop)
ode_cache = allocate_cache(ode_op,μ,t0)
vθ = similar(w)
vθ .= 0.0
l_cache = nothing
A,b = _allocate_matrix_and_vector(ode_op,μ,t0,vθ,ode_cache)
