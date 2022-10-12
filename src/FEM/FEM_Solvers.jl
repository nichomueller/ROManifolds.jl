function FE_solve(
  FEMInfo::FOMInfoS,
  operator::AffineFEOperator)

  if FEMInfo.solver == "lu"
    uₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field = solve(LinearFESolver(), operator)
  end

  get_free_dof_values(uₕ_field)

end

function FE_solve(
  FEMSpace::FOMPoissonST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  function get_uₕ(uₕ_field)
    uₕ, _ = uₕ_field
    get_free_dof_values(uₕ)::Vector{Float}
  end

  m(t, u, v) = ∫(Param.m(t)*(u*v)) * FEMSpace.dΩ
  a(t, u, v) = ∫(∇(v)⋅(Param.α(t)*∇(u)))*FEMSpace.dΩ
  rhs(t, v) = ∫(v*Param.f(t))*FEMSpace.dΩ + ∫(v*Param.h(t))*FEMSpace.dΓn
  operator = TransientAffineFEOperator(m, a, rhs, FEMSpace.V, FEMSpace.V₀)

  linear_solver = LUSolver()
  ode_solver = ThetaMethod(linear_solver, FEMInfo.δt, FEMInfo.θ)

  u₀_field = interpolate_everywhere(Param.u₀, FEMSpace.V(FEMInfo.t₀))

  uₕₜ_field = solve(ode_solver, operator, u₀_field, FEMInfo.t₀, FEMInfo.tₗ)
  uₕₜ = Broadcasting(get_uₕ)(uₕₜ_field)

  blocks_to_matrix(uₕₜ)

end

function FE_solve(
  FEMSpace::FOMStokesS,
  FEMInfo::FOMInfoS,
  Param::ParamInfoS)

  a((u,p),(v,q)) = ∫( ∇(v)⊙(Param.α*∇(u)) - Param.b*((∇⋅v)*p + q*(∇⋅u)) ) * FEMSpace.dΩ
  rhs((v,q)) = ∫(v ⋅ Param.f) * FEMSpace.dΩ + ∫(v ⋅ Param.h) * FEMSpace.dΓn
  operator = AffineFEOperator(a, rhs, FEMSpace.X, FEMSpace.X₀)

  if FEMInfo.solver == "lu"
    uₕ_field, pₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field, pₕ_field = solve(LinearFESolver(), operator)
  end

  Vector(get_free_dof_values(uₕ_field)), Vector(get_free_dof_values(pₕ_field))

end

function FE_solve(
  FEMSpace::FOMStokesST,
  FEMInfo::FOMInfoST,
  Param::ParamInfoST)

  m(t,(u,p),(v,q)) = ∫(Param.m(t)*(u⋅v)) * FEMSpace.dΩ
  a(t,(u,p),(v,q)) = ∫( ∇(v)⊙(Param.α(t)*∇(u)) - Param.b(t)*((∇⋅v)*p + q*(∇⋅u)) ) * FEMSpace.dΩ
  rhs(t,(v,q)) = ∫(v ⋅ Param.f(t)) * FEMSpace.dΩ + ∫(v ⋅ Param.h(t)) * FEMSpace.dΓn
  operator = TransientAffineFEOperator(m, a, rhs, FEMSpace.X, FEMSpace.X₀)

  linear_solver = LUSolver()
  ode_solver = ThetaMethod(linear_solver, FEMInfo.δt, FEMInfo.θ)

  u₀_field = interpolate_everywhere(Param.x₀(0.)[1], FEMSpace.V(FEMInfo.t₀))
  p₀_field = interpolate_everywhere(Param.x₀(0.)[2], FEMSpace.Q(FEMInfo.t₀))
  x₀_field = interpolate_everywhere([u₀_field, p₀_field], FEMSpace.X(FEMInfo.t₀))

  xₕₜ_field = solve(ode_solver, operator, x₀_field, FEMInfo.t₀, FEMInfo.tₗ)
  uₕₜ = zeros(FEMSpace.Nₛᵘ, convert(Int, FEMInfo.tₗ / FEMInfo.δt))
  pₕₜ = zeros(FEMSpace.Nₛᵖ, convert(Int, FEMInfo.tₗ / FEMInfo.δt))
  count = 1
  for (xₕ, _) in xₕₜ_field
    uₕₜ[:, count] = get_free_dof_values(xₕ[1])
    pₕₜ[:, count] = get_free_dof_values(xₕ[2])
    count += 1
  end

  uₕₜ, pₕₜ

end

function FE_solve(
  FEMSpace::FOMNavierStokesS,
  ::FOMInfoS,
  Param::ParamInfoS) where T

  a((u,p),(v,q)) = ∫( ∇(v)⊙(Param.α*∇(u)) - Param.b*(∇⋅v)*p + q*(∇⋅u) ) * FEMSpace.dΩ

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) ) * FEMSpace.dΩ
  dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) ) * FEMSpace.dΩ

  rhs((v,q)) = ∫(v ⋅ Param.f) * FEMSpace.dΩ + ∫(v ⋅ Param.h) * FEMSpace.dΓn

  res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v) - rhs((v,q))
  jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

  operator = FEOperator(res, jac, FEMSpace.X, FEMSpace.X₀)

  nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
  solver = FESolver(nls)
  uₕ_field, pₕ_field = solve(solver, operator)

  Vector(get_free_dof_values(uₕ_field)), Vector(get_free_dof_values(pₕ_field))

end
