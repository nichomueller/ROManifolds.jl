include("../utils/general.jl")

function get_FE_space(probl::ProblemSpecifics, model::UnstructuredDiscreteModel, g = nothing)

  degree = 2 .* probl.order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γ = BoundaryTriangulation(model, tags=probl.neumann_tags)
  dΓ = Measure(Γ, degree)
  Qₕ = CellQuadrature(Ω, degree)

  labels = get_face_labeling(model)
  if !isempty(probl.dirichlet_tags) && !isempty(probl.dirichlet_labels)
    for i = 1:length(probl.dirichlet_tags)
      add_tag_from_tags!(labels, probl.dirichlet_tags[i], probl.dirichlet_labels[i])
    end
  end

  ref_FE = ReferenceFE(lagrangian, Float64, probl.order)
  V₀ = TestFESpace(model, ref_FE; conformity=:H1, dirichlet_tags=probl.dirichlet_tags, labels=labels)
  if !isnothing(g)
    V = TrialFESpace(V₀, g)
  else
    V = TrialFESpace(V₀, (x -> 0))
  end

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  σₖ = get_cell_dof_ids(V₀)
  Nₕ = length(get_free_dof_ids(V))

  FE_space = FESpacePoisson(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, Ω, dΩ, dΓ)

  return FE_space

end

function get_FE_space(probl::ProblemSpecificsUnsteady, model::UnstructuredDiscreteModel, g = nothing)

  Ω = Interior(model)
  degree = 2 .* probl.order
  dΩ = Measure(Ω, degree)
  Γ = BoundaryTriangulation(model, tags=probl.neumann_tags)
  dΓ = Measure(Γ, degree)
  Qₕ = CellQuadrature(Ω, degree)

  labels = get_face_labeling(model)
  if !isempty(probl.dirichlet_tags) && !isempty(probl.dirichlet_labels)
    for i = 1:length(probl.dirichlet_tags)
      add_tag_from_tags!(labels, probl.dirichlet_tags[i], probl.dirichlet_labels[i])
    end
  end

  ref_FE = ReferenceFE(lagrangian, Float64, probl.order)
  V₀ = TestFESpace(model, ref_FE; conformity=:H1, dirichlet_tags=probl.dirichlet_tags, labels=labels)
  if isnothing(g)
    g₀(x, t::Real) = 0
    g₀(t::Real) = x->g₀(x,t)
    V = TransientTrialFESpace(V₀, g₀)
  else
    V = TransientTrialFESpace(V₀, g)
  end

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ(t) = get_trial_fe_basis(V(t))
  σₖ = get_cell_dof_ids(V₀)
  Nₕ = length(get_free_dof_ids(V₀))

  FE_space = FESpacePoissonUnsteady(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, Ω, dΩ, dΓ)

  return FE_space

end

function assemble_mass(FE_space::SteadyProblem, probl::SteadyProblem, param::ParametricSpecifics)

  if !probl.probl_nl["M"]
    return assemble_matrix(∫(FE_space.ϕᵥ * FE_space.ϕᵤ) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  else
    return assemble_matrix(∫(FE_space.ϕᵥ * (param.m * FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  end

end

function assemble_mass(FE_space::UnsteadyProblem, probl::UnsteadyProblem, param::ParametricSpecificsUnsteady)

  function unsteady_mass(t)
    if !probl.probl_nl["M"]
      return assemble_matrix(∫(FE_space.ϕᵥ * FE_space.ϕᵤ(t)) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    else
      return assemble_matrix(∫(FE_space.ϕᵥ * (param.m(t) * FE_space.ϕᵤ(t))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    end
  end

  return unsteady_mass

end

function assemble_stiffness(FE_space::SteadyProblem, probl::SteadyProblem, param::ParametricSpecifics)

  if !probl.probl_nl["A"]
    A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  else
    A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α * ∇(FE_space.ϕᵤ))) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  end

  return A

end

function assemble_stiffness(FE_space::UnsteadyProblem, probl::UnsteadyProblem, param::ParametricSpecificsUnsteady)

  function unsteady_stiffness(t)
    if !probl.probl_nl["A"]
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ(t))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    else
      return assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α(t) * ∇(FE_space.ϕᵤ(t)))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)
    end
  end

  return unsteady_stiffness

end

function assemble_forcing(FE_space::SteadyProblem, probl::SteadyProblem, param::ParametricSpecifics)

  if !probl.probl_nl["f"] && !probl.probl_nl["h"]
    return assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΩ, FE_space.V₀), assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΓ, FE_space.V₀)
  elseif !probl.probl_nl["f"] && probl.probl_nl["h"]
    return assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΩ, FE_space.V₀), assemble_vector(∫(FE_space.ϕᵥ * param.h) * FE_space.dΓ, FE_space.V₀)
  elseif probl.probl_nl["f"] && !probl.probl_nl["h"]
    return assemble_vector(∫(FE_space.ϕᵥ * param.f) * FE_space.dΩ, FE_space.V₀), assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΓ, FE_space.V₀)
  elseif probl.probl_nl["f"] && probl.probl_nl["h"]
    return assemble_vector(∫(FE_space.ϕᵥ * param.f) * FE_space.dΩ, FE_space.V₀), assemble_vector(∫(FE_space.ϕᵥ * param.h) * FE_space.dΓ, FE_space.V₀)
  end

end

function assemble_forcing(FE_space::UnsteadyProblem, probl::UnsteadyProblem, param::ParametricSpecificsUnsteady)

  function unsteady_forcing(t)
    if !probl.probl_nl["f"] && !probl.probl_nl["h"]
      return assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΩ, FE_space.V₀), assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΓ, FE_space.V₀)
    elseif !probl.probl_nl["f"] && probl.probl_nl["h"]
      return assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΩ, FE_space.V₀), assemble_vector(∫(FE_space.ϕᵥ * param.h(t)) * FE_space.dΓ, FE_space.V₀)
    elseif probl.probl_nl["f"] && !probl.probl_nl["h"]
      return assemble_vector(∫(FE_space.ϕᵥ * param.f(t)) * FE_space.dΩ, FE_space.V₀), assemble_vector(∫(FE_space.ϕᵥ) * FE_space.dΓ, FE_space.V₀)
    elseif probl.probl_nl["f"] && probl.probl_nl["h"]
      return assemble_vector(∫(FE_space.ϕᵥ * param.f(t)) * FE_space.dΩ, FE_space.V₀), assemble_vector(∫(FE_space.ϕᵥ * param.h(t)) * FE_space.dΓ, FE_space.V₀)
    end
  end

  return unsteady_forcing

end

function assemble_H1_norm_matrix(FE_space::SteadyProblem)

  Xᵘ = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀) +
  assemble_matrix(∫(FE_space.ϕᵥ * FE_space.ϕᵤ) * FE_space.dΩ, FE_space.V, FE_space.V₀)

  return Xᵘ

end

function assemble_H1_norm_matrix(FE_space::UnsteadyProblem)

  Xᵘ(t) = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ(t))) * FE_space.dΩ, FE_space.V(t), FE_space.V₀) +
  assemble_matrix(∫(FE_space.ϕᵥ * FE_space.ϕᵤ(t)) * FE_space.dΩ, FE_space.V(t), FE_space.V₀)

  return Xᵘ(0.0)

end

function assemble_H1_norm_matrix_nobcs(FE_space₀::SteadyProblem)

  Xᵘ₀ = assemble_matrix(∫(∇(FE_space₀.ϕᵥ) ⋅ ∇(FE_space₀.ϕᵤ)) * FE_space₀.dΩ, FE_space₀.V, FE_space₀.V₀) +
  assemble_matrix(∫(FE_space₀.ϕᵥ * FE_space₀.ϕᵤ) * FE_space₀.dΩ, FE_space₀.V, FE_space₀.V₀)

  return Xᵘ₀

end

function assemble_H1_norm_matrix_nobcs(FE_space₀::UnsteadyProblem)

  Xᵘ₀(t) = assemble_matrix(∫(∇(FE_space₀.ϕᵥ) ⋅ ∇(FE_space₀.ϕᵤ(t))) * FE_space₀.dΩ, FE_space₀.V(t), FE_space₀.V₀) +
  assemble_matrix(∫(FE_space₀.ϕᵥ * FE_space₀.ϕᵤ(t)) * FE_space₀.dΩ, FE_space₀.V(t), FE_space₀.V₀)

  return Xᵘ₀(0.0)

end

function FE_solve(FE_space::FESpacePoisson, probl::SteadyProblem, param::ParametricSpecifics; subtract_Ddata = true)

  _, Gₕ = get_lifting_operator(FE_space, param)

  a(u, v) = ∫(∇(v) ⋅ (param.α * ∇(u))) * FE_space.dΩ
  b(v) = ∫(v * param.f) * FE_space.dΩ + ∫(v * param.h) * FE_space.dΓ
  operator = AffineFEOperator(a, b, FE_space.V, FE_space.V₀)

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

  _, Gₕₜ = get_lifting_operator(FE_space, probl, param)

  m(t, u, v) = ∫( u * v )dΩ
  a(t, u, v) = ∫(∇(v) ⋅ (param.α(t) * ∇(u))) * FE_space.dΩ
  b(t, v) = ∫(v * param.f(t)) * FE_space.dΩ + ∫(v * param.h(t)) * FE_space.dΓ
  operator = TransientAffineFEOperator(m, a, b, FE_space.V, FE_space.V₀)

  linear_solver = LUSolver()

  if probl.time_method === "θ-method"
    ode_solver = ThetaMethod(linear_solver, probl.δt, probl.θ)
  else
    ode_solver = RungeKutta(linear_solver, probl.δt, probl.RK_type)
  end

  u₀_field = interpolate_everywhere(param.u₀, FE_space.V(probl.t₀))

  uₕₜ_field = solve(ode_solver, operator, u₀_field, probl.t₀, probl.T)
  uₕₜ = zeros(FE_space.Nₕ, convert(Int64, probl.T / probl.δt))
  global count = 0
  dΩ = FE_space.dΩ
  for (uₕ, _) in uₕₜ_field
    global count += 1
    uₕₜ[:, count] = get_free_dof_values(uₕ)
  end

  if subtract_Ddata
    uₕₜ = uₕₜ - Gₕₜ
  else
    uₕₜ = uₕₜ
  end

  return uₕₜ, Gₕₜ

end

#= function FE_solve_temp(FE_space::FESpacePoissonUnsteady, probl::ProblemSpecificsUnsteady, param::ParametricSpecificsUnsteady; subtract_Ddata = true)

  _, Gₕₜ = get_lifting_operator(FE_space, probl, param)

  F = assemble_forcing(FE_space, param)
  A = assemble_stiffness(FE_space, probl, param)
  M = assemble_mass(FE_space)
  uₕₜ = zeros(FE_space.Nₕ, convert(Int64, probl.T / probl.δt + 1))
  u₀_field = interpolate_everywhere(param.u₀, FE_space.V(probl.t₀))
  uₕₜ[:, 1] = get_free_dof_values(u₀_field)
  tᵢ_prev = probl.t₀

  if probl.case === 0
    AA = A(0.0)
    for (i, tᵢ) in enumerate(probl.t₀+probl.δt:probl.δt:probl.T)
      LHSⁿ = M + param.α(tᵢ)(Point(0.,0.))*AA*probl.δt/2
      LHS_prev = M - param.α(tᵢ_prev)(Point(0.,0.))*AA*probl.δt/2
      RHSⁿ = (F(tᵢ) + F(tᵢ_prev))*probl.δt/2
      uₕₜ[:, i+1] = LHSⁿ \ (LHS_prev*uₕₜ[:, i] + RHSⁿ)
      tᵢ_prev = tᵢ
    end
  else
    for (i, tᵢ) in enumerate(probl.t₀+probl.δt:probl.δt:probl.T)
      LHSⁿ = M + A(tᵢ)*probl.δt/2
      LHS_prev = M - A(tᵢ_prev)*probl.δt/2
      RHSⁿ = (F(tᵢ) + F(tᵢ_prev))*probl.δt/2
      uₕₜ[:, i+1] = LHSⁿ \ (LHS_prev*uₕₜ[:, i] + RHSⁿ)
      tᵢ_prev = tᵢ
    end
  end

  if subtract_Ddata
    uₕₜ = uₕₜ[:, 2:end] - Gₕₜ
  else
    uₕₜ = uₕₜ[:, 2:end]
  end

  return uₕₜ, Gₕₜ

end =#

#= function FE_solve(FE_space::FESpacePoisson, param::ParametricSpecificsUnsteady; subtract_Ddata = true)

_, Gₕ = get_lifting_operator(FE_space, param)

res(u,v) = ∫( ∇(v) ⊙ (param.α∘u ⋅ ∇(u)) - v * f) * FE_space.dΩ - ∫(v * param.h) * FE_space.dΓ
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

function get_lifting_operator(FE_space::FESpacePoisson, param::ParametricSpecifics)

  gₕ = interpolate_everywhere(param.g, FE_space.V)
  Gₕ = get_free_dof_values(gₕ)

  gₕ, Gₕ

end

function get_lifting_operator(FE_space::FESpacePoissonUnsteady, probl::ProblemSpecificsUnsteady, param::ParametricSpecificsUnsteady)

  gₕ(t) = interpolate_everywhere(param.g(t), FE_space.V(t))
  Gₕ = zeros(FE_space.Nₕ, convert(Int64, probl.T / probl.δt))
  for (i, tᵢ) in enumerate(probl.t₀+probl.δt:probl.δt:probl.T)
    Gₕ[:, i] = get_free_dof_values(gₕ(tᵢ))
  end

  gₕ, Gₕ

end
