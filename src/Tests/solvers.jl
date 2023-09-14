op,solver = feop,fesolver
ode_op = get_algebraic_operator(op)
K = 2
μ = realization(op,K)
t = solver.t0
w = PTArray([zeros(test.nfree) for _ = 1:K])
uF = copy(w)
times = get_times(solver)

# all steps
cache = nothing
results = PTArray[]

for t in times
  _,cache = solve_step!(uF,solver,ode_op,μ,w,t,cache)
  w = copy(uF)
  push!(results,w)
  println(get_free_dof_values(cache[1]))
end
# _,cache_new = solve_step!(uF,solver,ode_op,μ,w,t,cache)
# w = copy(uF)
# push!(results,w)

# _,cache_new_new = solve_step!(uF,solver,ode_op,μ,w,t+dt,cache_new)
# w = copy(uF)
# cache = copy(cache_new)
# push!(results,w)

# _,cache_new = solve_step!(uF,solver,ode_op,μ,w,t+2dt,cache)
# w = copy(w)
# cache = copy(cache_new)
# push!(results,uF)


# Gridap
p = μ[1]
gok(x,t) = g(x,p,t)
gok(t) = x->gok(x,t)
mok(t,u,v) = ∫(v*u)dΩ
aok(t,u,v) = ∫(a(p,t)*∇(v)⋅∇(u))dΩ
lok(t,v) = ∫(f(p,t)*v)dΩ + ∫(h(p,t)*v)dΓn
trial_ok = TransientTrialFESpace(test,gok)
feop_ok = TransientAffineFEOperator(mok,aok,lok,trial_ok,test)
ode_op_ok = Gridap.ODEs.TransientFETools.get_algebraic_operator(feop_ok)
solver_ok = ThetaMethod(LUSolver(),solver.dt,solver.θ)
u0_ok = interpolate_everywhere(u0(p),trial_ok(t0))
uht_ok = solve(solver_ok,feop_ok,u0_ok,solver.t0,solver.tF)

# all steps
results_ok = Vector{Float}[]
for (uh,t) in uht_ok
  u_ok = copy(get_free_dof_values(uh))
  push!(results_ok,u_ok)
end

for (res,res_ok) in zip(results,results_ok)
  test_ptarray(res,res_ok)
end

# 1st step
run_1st_step = true
if run_1st_step
  w = PTArray([zeros(test.nfree) for _ = 1:K])
  uF = copy(w)
  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t+dtθ

  ode_cache = allocate_cache(ode_op,μ)
  vθ = similar(w)
  vθ .= 0.0
  l_cache = nothing
  A,b = _allocate_matrix_and_vector(ode_op,w,ode_cache)

  ode_cache = update_cache!(ode_cache,ode_op,μ,tθ)

  _matrix_and_vector!(A,b,ode_op,μ,tθ,dtθ,w,ode_cache,vθ)
  afop = PAffineOperator(A,b)

  l_cache = solve!(uF,solver.nls,afop,l_cache)
  uF = uF + w

  # Gridap
  w_ok = zeros(test.nfree)
  uF_ok = copy(w_ok)
  dt_ok = solver_ok.dt
  solver_ok.θ == 0.0 ? dtθ_ok = dt_ok : dtθ_ok = dt_ok*solver_ok.θ
  tθ_ok = t+dtθ_ok

  ode_cache_ok = Gridap.ODEs.TransientFETools.allocate_cache(ode_op_ok)
  vθ_ok = similar(w_ok)
  vθ_ok .= 0.0
  l_cache_ok = nothing
  A_ok,b_ok = Gridap.ODEs.ODETools._allocate_matrix_and_vector(ode_op_ok,w_ok,ode_cache_ok)

  ode_cache_ok = Gridap.ODEs.TransientFETools.update_cache!(ode_cache_ok,ode_op_ok,tθ)

  Gridap.ODEs.ODETools._matrix_and_vector!(A_ok,b_ok,ode_op_ok,tθ_ok,dtθ_ok,w_ok,ode_cache_ok,vθ_ok)
  afop_ok = AffineOperator(A_ok,b_ok)

  l_cache_ok = solve!(uF_ok,solver_ok.nls,afop_ok,l_cache_ok)
  uF_ok = uF_ok + w_ok

  test_ptarray(A,A_ok)
  test_ptarray(b,b_ok)
  test_ptarray(uF,uF_ok)
end


################################################################################


op = feop
ode_op = get_algebraic_operator(op)
K = 2
μ = realization(op,K)
t = t0
w = PTArray([zeros(test.nfree) for _ = 1:K])

ode_solver = ThetaMethod(LUSolver(),dt,θ)
sol = MySol(ode_solver,ode_op,μ,w,t0,tF)

new_results = PTArray[]
for (uh,t) in sol
  ye = copy(uh)
  push!(new_results,ye)
end
