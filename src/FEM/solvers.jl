function FE_solve(FE_space::FESpacePoisson, probl::SteadyProblem, param::ParametricSpecifics; subtract_Ddata = true)

  Gₕ = assemble_lifting(FE_space, param)

  a(u, v) = ∫(∇(v) ⋅ (param.α * ∇(u))) * FE_space.dΩ
  rhs(v) = ∫(v * param.f) * FE_space.dΩ + ∫(v * param.h) * FE_space.dΓn
  operator = AffineFEOperator(a, rhs, FE_space.V, FE_space.V₀)

  if probl.solver === "lu"
    uₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field = solve(LinearFESolver(), operator)
  end

  if subtract_Ddata
    uₕ = get_free_dof_values(uₕ_field) - Gₕ
  else
    uₕ = get_free_dof_values(uₕ_field)
  end

  return uₕ, Gₕ

end

function FE_solve(FE_space::FESpacePoissonUnsteady, probl::ProblemSpecificsUnsteady, param::ParametricSpecificsUnsteady; subtract_Ddata = true)

  Gₕₜ = assemble_lifting(FE_space, probl, param)

  m(t, u, v) = ∫(param.m(t) * ( u * v ))dΩ
  a(t, u, v) = ∫(∇(v) ⋅ (param.α(t) * ∇(u))) * FE_space.dΩ
  rhs(t, v) = ∫(v * param.f(t)) * FE_space.dΩ + ∫(v * param.h(t)) * FE_space.dΓn
  operator = TransientAffineFEOperator(m, a, rhs, FE_space.V, FE_space.V₀)

  linear_solver = LUSolver()

  if probl.time_method === "θ-method"
    ode_solver = ThetaMethod(linear_solver, probl.δt, probl.θ)
  else
    ode_solver = RungeKutta(linear_solver, probl.δt, probl.RK_type)
  end

  u₀_field = interpolate_everywhere(param.u₀, FE_space.V(probl.t₀))

  uₕₜ_field = solve(ode_solver, operator, u₀_field, probl.t₀, probl.T)
  uₕₜ = zeros(FE_space.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  global count = 0
  dΩ = FE_space.dΩ
  for (uₕ, _) in uₕₜ_field
    global count += 1
    uₕₜ[:, count] = get_free_dof_values(uₕ)
  end

  if subtract_Ddata
    uₕₜ -= Gₕₜ
  end

  return uₕₜ, Gₕₜ

end

function FE_solve(FE_space::FESpaceStokesUnsteady, probl::ProblemSpecificsUnsteady, param::ParametricSpecificsUnsteady; subtract_Ddata = true)

  Gₕₜ = assemble_lifting(FE_space, probl, param)

  m(t, (u, p), (v, q)) = ∫(param.m(t) * ( u ⋅ v )) * FE_space.dΩ
  ab(t, (u, p), (v, q)) = ∫(param.α(t)*(∇(v) ⊙ ∇(u))) * FE_space.dΩ - ∫((∇⋅v) * p) * FE_space.dΩ + ∫(q * (∇⋅u)) * FE_space.dΩ
  rhs(t, (v, q)) = ∫(v ⋅ param.f(t)) * FE_space.dΩ + ∫(v ⋅ param.h(t)) * FE_space.dΓn
  operator = TransientAffineFEOperator(m, ab, rhs, FE_space.X, FE_space.X₀)

  linear_solver = LUSolver()

  if probl.time_method === "θ-method"
    ode_solver = ThetaMethod(linear_solver, probl.δt, probl.θ)
  else
    ode_solver = RungeKutta(linear_solver, probl.δt, probl.RK_type)
  end

  u0(x) = param.u₀(x)[1]
  p0(x) = param.u₀(x)[2]
  u₀_field = interpolate_everywhere(u0, FE_space.V(probl.t₀))
  p₀_field = interpolate_everywhere(p0, FE_space.Q(probl.t₀))
  x₀_field = interpolate_everywhere([u₀_field,p₀_field], FE_space.X(probl.t₀))

  xₕₜ_field = solve(ode_solver, operator, x₀_field, probl.t₀, probl.T)
  uₕₜ = zeros(FE_space.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  pₕₜ = zeros(FE_space.Nₛᵖ, convert(Int64, probl.T / probl.δt))
  global count = 0
  dΩ = FE_space.dΩ
  for (xₕ, _) in xₕₜ_field
    global count += 1
    uₕₜ[:, count] = get_free_dof_values(xₕ[1])
    pₕₜ[:, count] = get_free_dof_values(xₕ[2])
  end

  if subtract_Ddata
    uₕₜ -= Gₕₜ
  end

  return uₕₜ, Gₕₜ

end

function check_stokes_solver()
  A = assemble_stiffness(FE_space, problem_info, parametric_info)(0.0)
  M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
  Bᵀ = assemble_primal_opᵀ(FE_space)(0.0)
  B = assemble_primal_opᵀ(FE_space)(0.0)
  F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
  H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)

  δt = 0.005
  θ = 0.5

  u1 = uₕₜ[:,1]
  p1 = pₕₜ[:,1]
  res1 = θ*(δt*θ*A*parametric_info.α + M)*u1 + δt*θ*Bᵀ*p1 - δt*θ*(F+H)
  res2 = B*u1

  u2 = uₕₜ[:,1]
  p2 = pₕₜ[:,1]
  res1 = θ*(δt*θ*A*parametric_info.α + M)*u2 + ((1-θ)*δt*θ*A*parametric_info.α - θ*M)*u1 + δt*θ*Bᵀ*p2 - δt*θ*(F+H)
  res2 = B*u2

end

#= function FE_solve(FE_space::FESpacePoisson, param::ParametricSpecificsUnsteady; subtract_Ddata = true)

_, Gₕ = get_lifting_operator(FE_space, param)

res(u,v) = ∫( ∇(v) ⊙ (param.α∘u ⋅ ∇(u)) - v * f) * FE_space.dΩ - ∫(v * param.h) * FE_space.dΓn
jac(u, du, v) = ∫( ∇(v) ⊙ (param.dα∘(du, ∇(u))) )*dΩ
operator = FEOperator(res, jac, FE_space.V, FE_space.V₀)

nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

initial_uₕ_guess = FEFunction(FE_space.V, rand(Float64, num_free_dofs(FE_space.V)))
uₕ_field, _ = solve!(initial_uₕ_guess, solver, operator)

if subtract_Ddata
  uₕ = get_free_dof_values(uₕ_field) - Gₕ
else
  uₕ = get_free_dof_values(uₕ_field)
end

return uₕ, Gₕ

end =#

function assemble_lifting(FE_space::SteadyProblem, param::ParametricSpecifics)

  gₕ = interpolate_everywhere(param.g, FE_space.V)
  Gₕ = get_free_dof_values(gₕ)

  Gₕ

end

function assemble_lifting(FE_space::UnsteadyProblem, probl::ProblemSpecificsUnsteady, param::ParametricSpecificsUnsteady)

  gₕ(t) = interpolate_everywhere(param.g(t), FE_space.V(t))
  Gₕ = zeros(FE_space.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  for (i, tᵢ) in enumerate(probl.t₀+probl.δt:probl.δt:probl.T)
    Gₕ[:, i] = get_free_dof_values(gₕ(tᵢ))
  end

  Gₕ

end
