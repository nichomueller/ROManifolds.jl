K = 2
μ = realization(feop,K)
ode_solver = ThetaMethod(LUSolver(),dt,θ)

# POISSON
op = feop
ode_op = get_algebraic_operator(op)
w = get_free_dof_values(uh0μ(μ))
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

# STOKES
op = feop
ode_op = get_algebraic_operator(op)
w = get_free_dof_values(xh0μ(μ))
sol = PODESolution(ode_solver,ode_op,μ,w,t0,tF)

results = PTArray[]
for (uh,t) in sol
  ye = copy(uh)
  push!(results,ye)
end

g_ok(x,t) = g(x,μ[1],t)
g_ok(t) = x->g_ok(x,t)
u0_ok(x) = u0(x,μ[1])
p0_ok(x) = p0(x,μ[1])
a_ok(t,(u,p),(v,q)) = ∫(a(μ[1],t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫ₚ(q*(∇⋅(u)))dΩ
b_ok(t,(v,q)) = ∫(v⋅f(μ[1],t))dΩ + ∫(v⋅h(μ[1],t))dΓn
m_ok(t,(ut,pt),(v,q)) = ∫(v⋅ut)dΩ

trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok =  TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
uh0_ok = interpolate_everywhere(u0_ok,trial_u_ok(t0))
ph0_ok = interpolate_everywhere(p0_ok,trial_p(t0))
xh0_ok = interpolate_everywhere([uh0_ok,ph0_ok],trial_ok(t0))
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
