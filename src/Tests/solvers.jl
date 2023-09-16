op = feop
ode_op = get_algebraic_operator(op)
K = 2
μ = realization(op,K)
t = t0
w = PTArray([zeros(test.nfree) for _ = 1:K])

ode_solver = ThetaMethod(LUSolver(),dt,θ)
sol = PODESolution(ode_solver,ode_op,μ,w,t0,tF)

results = PTArray[]
for (uh,t) in sol
  ye = copy(uh)
  push!(results,ye)
end

g_ok(x,t) = g(x,μ[1],t)
g_ok(t) = x->g_ok(x,t)
a_ok(t,u,v) = ∫(a(μ[1],t)*∇(v)⋅∇(u))dΩ
b_ok(t,v) = ∫(v*f(μ[1],t))dΩ + ∫(v*h(μ[1],t))dΓn
m_ok(t,ut,v) = ∫(ut*v)dΩ

trial_ok = TransientTrialFESpace(test,g_ok)
feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,w[1],t0,tF)

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
