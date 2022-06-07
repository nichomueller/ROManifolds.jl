function FE_solve(
  FEMSpace::FEMSpacePoissonSteady, probl::SteadyProblem,
  Param::ParametricInfoSteady; subtract_Ddata = true)

  Gₕ = assemble_lifting(FEMSpace, Param)

  a(u, v) = ∫(∇(v) ⋅ (Param.α * ∇(u))) * FEMSpace.dΩ
  rhs(v) = ∫(v * Param.f) * FEMSpace.dΩ + ∫(v * Param.h) * FEMSpace.dΓn
  operator = AffineFEOperator(a, rhs, FEMSpace.V, FEMSpace.V₀)

  if probl.solver == "lu"
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

function FE_solve(
  FEMSpace::FEMSpacePoissonUnsteady, probl::ProblemInfoUnsteady,
  Param::ParametricInfoUnsteady; subtract_Ddata = true)

  Gₕₜ = assemble_lifting(FEMSpace, probl, Param)

  m(t, u, v) = ∫(Param.m(t)*(u*v))dΩ
  a(t, u, v) = ∫(∇(v)⋅(Param.α(t)*∇(u)))*FEMSpace.dΩ
  rhs(t, v) = rhs_form(t,v,FEMSpace,Param)
  operator = TransientAffineFEOperator(m, a, rhs, FEMSpace.V, FEMSpace.V₀)

  linear_solver = LUSolver()

  if probl.time_method == "θ-method"
    ode_solver = ThetaMethod(linear_solver, probl.δt, probl.θ)
  else
    ode_solver = RungeKutta(linear_solver, probl.δt, probl.RK_type)
  end

  u₀_field = interpolate_everywhere(Param.u₀, FEMSpace.V(probl.t₀))

  uₕₜ_field = solve(ode_solver, operator, u₀_field, probl.t₀, probl.T)
  uₕₜ = zeros(FEMSpace.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  global count = 0
  dΩ = FEMSpace.dΩ
  for (uₕ, _) in uₕₜ_field
    global count += 1
    uₕₜ[:, count] = get_free_dof_values(uₕ)
  end

  if subtract_Ddata
    uₕₜ -= Gₕₜ
  end

  return uₕₜ, Gₕₜ

end

function FE_solve(
  FEMSpace::FEMSpaceStokesUnsteady, probl::ProblemInfoUnsteady,
  Param::ParametricInfoUnsteady; subtract_Ddata=false)

  R₁,R₂ = assemble_lifting(FEMSpace, probl, Param)

  m(t,(u,p),(v,q)) = ∫(Param.m(t)*(u⋅v))*FEMSpace.dΩ
  ab(t,(u,p),(v,q)) = ∫(Param.α(t)*(∇(v) ⊙ ∇(u)))*FEMSpace.dΩ - ∫((∇⋅v)*p)*FEMSpace.dΩ + ∫(q*(∇⋅u))*FEMSpace.dΩ
  rhs(t,(v,q)) = ∫(v⋅Param.f(t))*FEMSpace.dΩ + ∫(v⋅Param.h(t))*FEMSpace.dΓn + ∫(v⋅Param.f(t))*FEMSpace.dΩ + ∫(q*Param.g(t))*FEMSpace.dΩ - (R₁-R₂)
  operator = TransientAffineFEOperator(m, ab, rhs, FEMSpace.X, FEMSpace.X₀)

  linear_solver = LUSolver()

  if probl.time_method == "θ-method"
    ode_solver = ThetaMethod(linear_solver, probl.δt, probl.θ)
  else
    ode_solver = RungeKutta(linear_solver, probl.δt, probl.RK_type)
  end

  u0(x) = Param.u₀(x)[1]
  p0(x) = Param.u₀(x)[2]
  u₀_field = interpolate_everywhere(u0, FEMSpace.V(probl.t₀))
  p₀_field = interpolate_everywhere(p0, FEMSpace.Q(probl.t₀))
  x₀_field = interpolate_everywhere([u₀_field,p₀_field], FEMSpace.X(probl.t₀))

  xₕₜ_field = solve(ode_solver, operator, x₀_field, probl.t₀, probl.T)
  uₕₜ = zeros(FEMSpace.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  pₕₜ = zeros(FEMSpace.Nₛᵖ, convert(Int64, probl.T / probl.δt))
  global count = 0
  dΩ = FEMSpace.dΩ
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
  A = assemble_stiffness(FEMSpace, FEMInfo, Param)(0.0)
  M = assemble_mass(FEMSpace, FEMInfo, Param)(0.0)
  Bᵀ = assemble_primal_opᵀ(FEMSpace)(0.0)
  B = assemble_primal_opᵀ(FEMSpace)(0.0)
  F = assemble_forcing(FEMSpace, FEMInfo, Param)(0.0)
  H = assemble_neumann_datum(FEMSpace, FEMInfo, Param)(0.0)

  δt = 0.005
  θ = 0.5

  u1 = uₕₜ[:,1]
  p1 = pₕₜ[:,1]
  res1 = θ*(δt*θ*A*Param.α + M)*u1 + δt*θ*Bᵀ*p1 - δt*θ*(F+H)
  res2 = B*u1

  u2 = uₕₜ[:,1]
  p2 = pₕₜ[:,1]
  res1 = θ*(δt*θ*A*Param.α + M)*u2 + ((1-θ)*δt*θ*A*Param.α - θ*M)*u1 + δt*θ*Bᵀ*p2 - δt*θ*(F+H)
  res2 = B*u2

end

#= function FE_solve(FEMSpace::FEMSpacePoisson, Param::ParametricInfoUnsteady; subtract_Ddata = true)

_, Gₕ = get_lifting_operator(FEMSpace, Param)

res(u,v) = ∫( ∇(v) ⊙ (Param.α∘u ⋅ ∇(u)) - v * f) * FEMSpace.dΩ - ∫(v * Param.h) * FEMSpace.dΓn
jac(u, du, v) = ∫( ∇(v) ⊙ (Param.dα∘(du, ∇(u))) )*dΩ
operator = FEOperator(res, jac, FEMSpace.V, FEMSpace.V₀)

nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

initial_uₕ_guess = FEFunction(FEMSpace.V, rand(Float64, num_free_dofs(FEMSpace.V)))
uₕ_field, _ = solve!(initial_uₕ_guess, solver, operator)

if subtract_Ddata
  uₕ = get_free_dof_values(uₕ_field) - Gₕ
else
  uₕ = get_free_dof_values(uₕ_field)
end

return uₕ, Gₕ

end =#

function assemble_lifting(FEMSpace::SteadyProblem, Param::ParametricInfoSteady)

  Gₕ = zeros(FEMSpace.Nₛᵘ,1)
  if !isnothing(FEMSpace.dΓd)
    gₕ = interpolate_everywhere(Param.g, FEMSpace.V)
    Gₕ = get_free_dof_values(gₕ)
  end

  Gₕ

end

function assemble_lifting(
  FEMSpace::FEMSpacePoissonUnsteady, probl::ProblemInfoUnsteady,
  Param::ParametricInfoUnsteady)

  Gₕ = zeros(FEMSpace.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  if !isnothing(FEMSpace.dΓd)
    gₕ(t) = interpolate_everywhere(Param.g(t), FEMSpace.V(t))
    for (i, tᵢ) in enumerate(probl.t₀+probl.δt:probl.δt:probl.T)
      Gₕ[:, i] = get_free_dof_values(gₕ(tᵢ))
    end
  end

  Gₕ

end

function assemble_lifting(FEMSpace::FEMSpaceStokesUnsteady, ::ProblemInfoUnsteady, Param::ParametricInfoUnsteady)

  gₕ(t) = interpolate_dirichlet(Param.g(t), FEMSpace.V(t))
  R₁(t,v) = ∫(Param.α(t)*(∇(v) ⊙ ∇(gₕ(t))))*FEMSpace.dΓd
  R₂(t,q) = ∫(∇⋅(gₕ(t))*q)*FEMSpace.dΓd
  return R₁,R₂

end

function rhs_form(
  t::Real,v::FEBasis,FEMSpace::FEMSpacePoissonUnsteady,
  Param::ParametricInfoUnsteady)
  if !isnothing(FEMSpace.dΓn)
    return ∫(v*Param.f(t))*FEMSpace.dΩ + ∫(v*Param.h(t))*FEMSpace.dΓn
  else
    return ∫(v*Param.f(t))*FEMSpace.dΩ
  end
end
