function FE_solve(
  FESpace::FESpacePoisson, probl::SteadyProblem,
  Param::ParametricSpecifics; subtract_Ddata = true)

  Gₕ = assemble_lifting(FESpace, Param)

  a(u, v) = ∫(∇(v) ⋅ (Param.α * ∇(u))) * FESpace.dΩ
  rhs(v) = ∫(v * Param.f) * FESpace.dΩ + ∫(v * Param.h) * FESpace.dΓn
  operator = AffineFEOperator(a, rhs, FESpace.V, FESpace.V₀)

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
  FESpace::FESpacePoissonUnsteady, probl::ProblemSpecificsUnsteady,
  Param::ParametricSpecificsUnsteady; subtract_Ddata = true)

  Gₕₜ = assemble_lifting(FESpace, probl, Param)

  m(t, u, v) = ∫(Param.m(t)*(u*v))dΩ
  a(t, u, v) = ∫(∇(v)⋅(Param.α(t)*∇(u)))*FESpace.dΩ
  rhs(t, v) = rhs_form(t,v,FESpace,Param)
  operator = TransientAffineFEOperator(m, a, rhs, FESpace.V, FESpace.V₀)

  linear_solver = LUSolver()

  if probl.time_method == "θ-method"
    ode_solver = ThetaMethod(linear_solver, probl.δt, probl.θ)
  else
    ode_solver = RungeKutta(linear_solver, probl.δt, probl.RK_type)
  end

  u₀_field = interpolate_everywhere(Param.u₀, FESpace.V(probl.t₀))

  uₕₜ_field = solve(ode_solver, operator, u₀_field, probl.t₀, probl.T)
  uₕₜ = zeros(FESpace.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  global count = 0
  dΩ = FESpace.dΩ
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
  FESpace::FESpaceStokesUnsteady, probl::ProblemSpecificsUnsteady,
  Param::ParametricSpecificsUnsteady; subtract_Ddata=false)

  R₁,R₂ = assemble_lifting(FESpace, probl, Param)

  m(t,(u,p),(v,q)) = ∫(Param.m(t)*(u⋅v))*FESpace.dΩ
  ab(t,(u,p),(v,q)) = ∫(Param.α(t)*(∇(v) ⊙ ∇(u)))*FESpace.dΩ - ∫((∇⋅v)*p)*FESpace.dΩ + ∫(q*(∇⋅u))*FESpace.dΩ
  rhs(t,(v,q)) = ∫(v⋅Param.f(t))*FESpace.dΩ + ∫(v⋅Param.h(t))*FESpace.dΓn + ∫(v⋅Param.f(t))*FESpace.dΩ + ∫(q*Param.g(t))*FESpace.dΩ - (R₁-R₂)
  operator = TransientAffineFEOperator(m, ab, rhs, FESpace.X, FESpace.X₀)

  linear_solver = LUSolver()

  if probl.time_method == "θ-method"
    ode_solver = ThetaMethod(linear_solver, probl.δt, probl.θ)
  else
    ode_solver = RungeKutta(linear_solver, probl.δt, probl.RK_type)
  end

  u0(x) = Param.u₀(x)[1]
  p0(x) = Param.u₀(x)[2]
  u₀_field = interpolate_everywhere(u0, FESpace.V(probl.t₀))
  p₀_field = interpolate_everywhere(p0, FESpace.Q(probl.t₀))
  x₀_field = interpolate_everywhere([u₀_field,p₀_field], FESpace.X(probl.t₀))

  xₕₜ_field = solve(ode_solver, operator, x₀_field, probl.t₀, probl.T)
  uₕₜ = zeros(FESpace.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  pₕₜ = zeros(FESpace.Nₛᵖ, convert(Int64, probl.T / probl.δt))
  global count = 0
  dΩ = FESpace.dΩ
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
  A = assemble_stiffness(FESpace, FEMInfo, Param)(0.0)
  M = assemble_mass(FESpace, FEMInfo, Param)(0.0)
  Bᵀ = assemble_primal_opᵀ(FESpace)(0.0)
  B = assemble_primal_opᵀ(FESpace)(0.0)
  F = assemble_forcing(FESpace, FEMInfo, Param)(0.0)
  H = assemble_neumann_datum(FESpace, FEMInfo, Param)(0.0)

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

#= function FE_solve(FESpace::FESpacePoisson, Param::ParametricSpecificsUnsteady; subtract_Ddata = true)

_, Gₕ = get_lifting_operator(FESpace, Param)

res(u,v) = ∫( ∇(v) ⊙ (Param.α∘u ⋅ ∇(u)) - v * f) * FESpace.dΩ - ∫(v * Param.h) * FESpace.dΓn
jac(u, du, v) = ∫( ∇(v) ⊙ (Param.dα∘(du, ∇(u))) )*dΩ
operator = FEOperator(res, jac, FESpace.V, FESpace.V₀)

nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

initial_uₕ_guess = FEFunction(FESpace.V, rand(Float64, num_free_dofs(FESpace.V)))
uₕ_field, _ = solve!(initial_uₕ_guess, solver, operator)

if subtract_Ddata
  uₕ = get_free_dof_values(uₕ_field) - Gₕ
else
  uₕ = get_free_dof_values(uₕ_field)
end

return uₕ, Gₕ

end =#

function assemble_lifting(FESpace::SteadyProblem, Param::ParametricSpecifics)

  Gₕ = zeros(FESpace.Nₛᵘ,1)
  if !isnothing(FESpace.dΓd)
    gₕ = interpolate_everywhere(Param.g, FESpace.V)
    Gₕ = get_free_dof_values(gₕ)
  end

  Gₕ

end

function assemble_lifting(
  FESpace::FESpacePoissonUnsteady, probl::ProblemSpecificsUnsteady,
  Param::ParametricSpecificsUnsteady)

  Gₕ = zeros(FESpace.Nₛᵘ, convert(Int64, probl.T / probl.δt))
  if !isnothing(FESpace.dΓd)
    gₕ(t) = interpolate_everywhere(Param.g(t), FESpace.V(t))
    for (i, tᵢ) in enumerate(probl.t₀+probl.δt:probl.δt:probl.T)
      Gₕ[:, i] = get_free_dof_values(gₕ(tᵢ))
    end
  end

  Gₕ

end

function assemble_lifting(FESpace::FESpaceStokesUnsteady, probl::ProblemSpecificsUnsteady, Param::ParametricSpecificsUnsteady)

  gₕ(t) = interpolate_dirichlet(Param.g(t), FESpace.V(t))
  R₁(t,v) = ∫(Param.α(t)*(∇(v) ⊙ ∇(gₕ(t))))*FESpace.dΓd
  R₂(t,q) = ∫(∇⋅(gₕ(t))*q)*FESpace.dΓd
  return R₁,R₂

end

function rhs_form(
  t::Real,v::FEBasis,FESpace::FESpacePoissonUnsteady,
  Param::ParametricSpecificsUnsteady)
  if !isnothing(FESpace.dΓn)
    return ∫(v*Param.f(t))*FESpace.dΩ + ∫(v*Param.h(t))*FESpace.dΓn
  else
    return ∫(v*Param.f(t))*FESpace.dΩ
  end
end
