K = 2
μ = realization(feop,K)

# POISSON
ode_op = get_algebraic_operator(feop)
w = get_free_dof_values(uh0μ(μ))
sol = PODESolution(fesolver,ode_op,μ,w,t0,tf)

results = PTArray[]
for (uh,t) in sol
  ye = copy(uh)
  push!(results,ye)
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
  ye = copy(uh)
  push!(results_ok,ye)
end

for i in eachindex(results)
  test_ptarray(results[i],results_ok[i])
end

boh = PTArray[]
@time for (uh,t) in sol
  push!(boh,uh)
end

boh_ok = Vector{Float}[]
@time for (uh,t) in sol_gridap
  push!(boh_ok,uh)
end
