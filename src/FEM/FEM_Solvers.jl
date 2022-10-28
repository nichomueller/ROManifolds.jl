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

  ode_solver = ThetaMethod(LUSolver(), FEMInfo.δt, FEMInfo.θ)
  xₕₜ_field = solve(ode_solver, operator, x₀_field, FEMInfo.t₀, FEMInfo.tₗ)

  count = 1
  Nₛ = length(get_free_dof_values(x₀_field))
  Nₜ = Int((FEMInfo.tₗ - FEMInfo.t₀) / FEMInfo.δt)
  xₕₜ = Matrix{Float}(undef, Nₛ, Nₜ)
  for (xₕ, _) in xₕₜ_field
    println("Time step: $count")
    xₕₜ[:,count] = get_free_dof_values(xₕ)
    count += 1
  end

  xₕₜ::Matrix{Float}

end

function FEM_solver(
  FEMInfo::FOMInfoST{2},
  operator::TransientFEOperator,
  x₀_field::FEFunction)

  ode_solver = ThetaMethod(LUSolver(), FEMInfo.δt, FEMInfo.θ)
  xₕₜ_field = solve(ode_solver, operator, x₀_field, FEMInfo.t₀, FEMInfo.tₗ)

  count = 1
  Nₛᵘ = length(get_free_dof_values(x₀_field[1]))
  Nₛᵖ = length(get_free_dof_values(x₀_field[2]))
  Nₜ = Int((FEMInfo.tₗ - FEMInfo.t₀) / FEMInfo.δt)

  xₕₜ = zeros(Nₛᵘ+Nₛᵖ, Nₜ)
  count = 1
  for (xₕ, _) in xₕₜ_field
    println("Time step: $count")
    xₕₜ[:,count] = get_free_dof_values(xₕ)
    count += 1
  end

  (xₕₜ[1:Nₛᵘ,:], xₕₜ[Nₛᵘ+1:end,:])::NTuple{2,Matrix{Float}}

end

function FEM_solver(
  FEMInfo::FOMInfoST{3},
  operator::TransientFEOperator,
  x₀_field::FEFunction)

  nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
  ode_solver = ThetaMethod(nls, FEMInfo.δt, FEMInfo.θ)
  xₕₜ_field = solve(ode_solver, operator, x₀_field, FEMInfo.t₀, FEMInfo.tₗ)

  count = 1
  Nₛᵘ = length(get_free_dof_values(x₀_field[1]))
  Nₛᵖ = length(get_free_dof_values(x₀_field[2]))
  Nₜ = Int((FEMInfo.tₗ - FEMInfo.t₀) / FEMInfo.δt)

  xₕₜ = zeros(Nₛᵘ+Nₛᵖ, Nₜ)
  count = 1
  for (xₕ, _) in xₕₜ_field
    println("Time step: $count")
    xₕₜ[:,count] = get_free_dof_values(xₕ)
    count += 1
  end

  (xₕₜ[1:Nₛᵘ,:], xₕₜ[Nₛᵘ+1:end,:])::NTuple{2,Matrix{Float}}

end
