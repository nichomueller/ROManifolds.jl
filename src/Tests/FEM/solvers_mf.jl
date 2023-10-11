K = 2
μ = realization(feop,K)
ode_solver = fesolver

op = feop
ode_op = get_algebraic_operator(op)
w = get_free_dof_values(xh0μ(μ))
sol = PODESolution(ode_solver,ode_op,μ,w,t0,tf)

results = PTArray[]
for (uh,t) in sol
  ye = copy(uh)
  push!(results,ye)
end

g_ok(x,t) = g(x,μ[1],t)
g_ok(t) = x->g_ok(x,t)
u0_ok(x) = u0(x,μ[1])
p0_ok(x) = p0(x,μ[1])
a_ok(t,(u,p),(v,q)) = ∫(a(μ[1],t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
b_ok(t,(v,q)) = ∫(v⋅f(μ[1],t))dΩ
m_ok(t,(ut,pt),(v,q)) = ∫(v⋅ut)dΩ

trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok =  TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientAffineFEOperator(m_ok,a_ok,b_ok,trial_ok,test)
uh0_ok = interpolate_everywhere(u0_ok,trial_u_ok(t0))
ph0_ok = interpolate_everywhere(p0_ok,trial_p(t0))
xh0_ok = interpolate_everywhere([uh0_ok,ph0_ok],trial_ok(t0))
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
ode_solver = ThetaMethod(LUSolver(),dt,θ)
sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,w[1],t0,tf)

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

# NONLINEAR

# t0 = 0.
# ode_op = get_algebraic_operator(op)
# w0,wf = copy(w),copy(w)
# dt = solver.dt
# solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
# tθ = t0+dtθ
# ode_cache = allocate_cache(ode_op,μ)
# vθ = similar(w0)
# vθ .= 0.0
# l_cache = nothing
# A,b = _allocate_matrix_and_vector(ode_op,w0,ode_cache)
# ode_cache = update_cache!(ode_cache,ode_op,μ,tθ)
# _matrix_and_vector!(A,b,ode_op,μ,tθ,dtθ,w0,ode_cache,vθ)
# afop = PAffineOperator(A,b)
# l_cache = solve!(wf,solver.nls,afop,l_cache)
# wf = wf + w0
# w0.array .= wf.array
# t0 = t0+dt

# tθ = t0+dtθ
# ode_cache = update_cache!(ode_cache,ode_op,μ,tθ)
# _matrix_and_vector!(A,b,ode_op,μ,tθ,dtθ,w0,ode_cache,vθ)
# afop = PAffineOperator(A,b)
# l_cache = solve!(wf,solver.nls,afop,l_cache)
# wf = wf + w0
# w0.array .= wf.array
# t0 = t0+dt

# t0_ok = 0.
# w_ok = copy(w[1])
# ode_op_ok = get_algebraic_operator(feop_ok)
# w0_ok,wf_ok = copy(w_ok),copy(w_ok)
# dt_ok = solver.dt
# solver.θ == 0.0 ? dtθ_ok = dt_ok : dtθ_ok = dt_ok*solver.θ
# tθ_ok = t0_ok+dtθ_ok
# ode_cache_ok = Gridap.ODEs.ODETools.allocate_cache(ode_op_ok)
# vθ_ok = similar(w0_ok)
# vθ_ok .= 0.0
# l_cache_ok = nothing
# Aok,bok = Gridap.ODEs.ODETools._allocate_matrix_and_vector(ode_op_ok,w0_ok,ode_cache_ok)
# ode_cache_ok = Gridap.ODEs.ODETools.update_cache!(ode_cache_ok,ode_op_ok,tθ_ok)
# Gridap.ODEs.ODETools._matrix_and_vector!(Aok,bok,ode_op_ok,tθ_ok,dtθ_ok,w0_ok,ode_cache_ok,vθ_ok)
# afop_ok = AffineOperator(Aok,bok)
# l_cache_ok = solve!(wf_ok,solver.nls,afop_ok,l_cache_ok)
# wf_ok .+= w0_ok
# w0_ok .= wf_ok
# t0_ok = t0_ok+dt_ok

# tθ_ok = t0_ok+dtθ_ok
# ode_cache_ok = Gridap.ODEs.ODETools.update_cache!(ode_cache_ok,ode_op_ok,tθ_ok)
# Gridap.ODEs.ODETools._matrix_and_vector!(Aok,bok,ode_op_ok,tθ_ok,dtθ_ok,w0_ok,ode_cache_ok,vθ_ok)
# afop_ok = AffineOperator(Aok,bok)
# l_cache_ok = solve!(wf_ok,solver.nls,afop_ok,l_cache_ok)
# wf_ok .+= w0_ok
# w0_ok .= wf_ok
# t0_ok = t0_ok+dt_ok

# test_ptarray(w0,w0_ok)
# test_ptarray(wf,wf_ok)

m_ok(t,u,v) = ∫(v⋅u)dΩ
a_ok(t,(u,p),(v,q)) = ∫(a(μ[1],t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ
c_ok(t,u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
dc_ok(t,u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

jac_t_ok(t,(u,p),(dut,dpt),(v,q)) = m_ok(t,dut,v)
jac_ok(t,(u,p),(du,dp),(v,q)) = a_ok(t,(du,dp),(v,q)) + dc_ok(t,u,du,v)
res_ok(t,(u,p),(v,q)) = a_ok(t,(u,p),(v,q)) + c_ok(t,u,v)
g_ok(x,t) = g(x,μ[1],t)
g_ok(t) = x->g_ok(x,t)
u0_ok(x) = u0(x,μ[1])
p0_ok(x) = p0(x,μ[1])

trial_u_ok = TransientTrialFESpace(test_u,g_ok)
trial_ok =  TransientMultiFieldFESpace([trial_u_ok,trial_p])
feop_ok = TransientFEOperator(res_ok,jac_ok,jac_t_ok,trial_ok,test)
uh0_ok = interpolate_everywhere(u0_ok,trial_u_ok(t0))
ph0_ok = interpolate_everywhere(p0_ok,trial_p(t0))
xh0_ok = interpolate_everywhere([uh0_ok,ph0_ok],trial_ok(t0))
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
sol_gridap = Gridap.ODEs.TransientFETools.GenericODESolution(ode_solver,ode_op_ok,w[1],t0,tf)
# nls = NLSolver(show_trace=true,method=:newton)
# fesolver_ok = ThetaMethod(nls,dt,θ)
results_ok = Vector{Float}[]
for (uh,t) in sol_gridap
  ye = copy(uh)
  push!(results_ok,ye)
end

for i in eachindex(results)
  test_ptarray(results[i],results_ok[i])
end
