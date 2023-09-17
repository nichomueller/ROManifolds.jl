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

# check efficiency without solve
function solution_step!(
  uf::PTArray,
  solver::ThetaMethod,
  op::AffinePODEOperator,
  μ::AbstractVector,
  u0::PTArray,
  t0::Real,
  cache)
  # println("in the test step -- my code")
  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op,μ)
    vθ = similar(u0)
    vθ .= 0.0
    l_cache = nothing
    A,b = _allocate_matrix_and_vector(op,u0,ode_cache)
  else
    ode_cache,vθ,A,b,l_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  _matrix_and_vector!(A,b,op,μ,tθ,dtθ,u0,ode_cache,vθ)

  uf = uf + u0
  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,A,b,l_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

function solve_step!(
  uf::AbstractVector,
  solver::ThetaMethod,
  op::Gridap.ODEs.ODETools.AffineODEOperator,
  u0::AbstractVector,
  t0::Real,
  cache) # -> (uF,tF)
  # println("in the test step -- Gridap")
  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if cache === nothing
  ode_cache = Gridap.ODEs.TransientFETools.allocate_cache(op)
  vθ = similar(u0)
  vθ .= 0.0
  l_cache = nothing
  A, b = Gridap.ODEs.ODETools._allocate_matrix_and_vector(op,u0,ode_cache)
  else
  ode_cache, vθ, A, b, l_cache = cache
  end

  ode_cache = Gridap.ODEs.TransientFETools.update_cache!(ode_cache,op,tθ)

  Gridap.ODEs.ODETools._matrix_and_vector!(A,b,op,tθ,dtθ,u0,ode_cache,vθ)

  uf = uf + u0

  cache = (ode_cache, vθ, A, b, l_cache)

  tf = t0+dt
  return (uf,tf,cache)
end
