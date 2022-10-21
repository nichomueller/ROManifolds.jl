function FEM_solver(
  FEMInfo::FOMInfoS{1},
  operator::AffineFEOperator)

  if FEMInfo.solver == "lu"
    uₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field = solve(LinearFESolver(), operator)
  end

  get_free_dof_values(uₕ_field)::Vector{Float}

end

function FEM_solver(
  FEMInfo::FOMInfoS{2},
  operator::AffineFEOperator)

  if FEMInfo.solver == "lu"
    uₕ_field, pₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field, pₕ_field = solve(LinearFESolver(), operator)
  end

  Vector(get_free_dof_values(uₕ_field)), Vector(get_free_dof_values(pₕ_field))

end

function FEM_solver(
  ::FOMInfoS{3},
  operator::FEOperator)

  nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
  solver = FESolver(nls)
  uₕ_field, pₕ_field = solve(solver, operator)

  Vector(get_free_dof_values(uₕ_field)), Vector(get_free_dof_values(pₕ_field))

end

function FEM_solver(
  FEMInfo::FOMInfoST{1},
  operator::TransientFEOperator,
  x₀_field::FEFunction)

  function xₕᵢ!(xₕ, xₕₜ)
    xₕ[1] = get_free_dof_values(xₕₜ)
    nothing
  end

  ode_solver = ThetaMethod(LUSolver(), FEMInfo.δt, FEMInfo.θ)
  xₕₜ_field = solve(ode_solver, operator, x₀_field, FEMInfo.t₀, FEMInfo.tₗ)

  count = 1
  Nₛ = length(get_free_dof_values(x₀_field))
  Nₜ = Int((FEMInfo.tₗ - FEMInfo.t₀) / FEMInfo.δt)
  xₕₜ = Matrix{Float}(undef, Nₛ, Nₜ)
  for (xₕ, _) in xₕₜ_field
    println("Time step: $count")
    xₕₜ[:,count] = get_free_dof_values(xₕ)
    #xₕᵢ!(xₕ[:,count], xₕₜ)
    count += 1
  end

  xₕₜ::Matrix{Float}

end

#= function FEM_solver(
  FEMSpace::FOMStokesST,
  FEMInfo::FOMInfoST{2},
  Param::ParamInfoST)

  m(t,(u,p),(v,q)) = ∫(Param.m(t)*(u⋅v))FEMSpace.dΩ
  a(t,(u,p),(v,q)) = ∫( ∇(v)⊙(Param.α(t)*∇(u)) - Param.b(t)*((∇⋅v)*p + q*(∇⋅u)) )FEMSpace.dΩ
  rhs(t,(v,q)) = ∫(v ⋅ Param.f(t))FEMSpace.dΩ + ∫(v ⋅ Param.h(t))FEMSpace.dΓn
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

end =#
