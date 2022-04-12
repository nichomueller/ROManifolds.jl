include("../utils/general.jl")

function get_FE_space(probl::ProblemSpecifics, model::UnstructuredDiscreteModel, g = nothing)
  #=MODIFY
  =#

  degree = 2 .* probl.order
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γ = BoundaryTriangulation(model, tags=probl.neumann_tags)
  dΓ = Measure(Γ, degree)
  Qₕ = CellQuadrature(Ω, degree)

  ref_FE = ReferenceFE(lagrangian, Float64, probl.order)
  V₀ = TestFESpace(model, ref_FE; conformity=:H1, dirichlet_tags=probl.dirichlet_tags)
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
  #=MODIFY
  =#

  Ω = Interior(model)
  dΩ = Measure(Ω, degree)
  Γ = BoundaryTriangulation(model, tags=probl.neumann_tags)
  dΓ = Measure(Γ, degree)
  degree = 2 .* probl.order
  Qₕ = CellQuadrature(Ω, degree)

  ref_FE = ReferenceFE(lagrangian, Float64, probl.order)
  V₀ = TestFESpace(model, ref_FE; conformity=:H1, dirichlet_tags=probl.dirichlet_tags)
  if !isnothing(g)
    V = TransientTrialFESpace(V₀, g)
  else
    V = TransientTrialFESpace(V₀, (x, t) -> 0)
  end

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  σₖ = get_cell_dof_ids(V₀)
  Nₕ = length(get_free_dof_ids(V))

  FE_space = FESpacePoisson(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, Ω, dΩ, dΓ)

  return FE_space

end

function assemble_stiffness(FE_space::FEMProblem, probl::Problem, param::ParametricSpecifics)
  #=MODIFY
  =#

  if probl.problem_nonlinearities["A"] === false
    A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  else
    A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α * ∇(FE_space.ϕᵤ))) * FE_space.dΩ, FE_space.V, FE_space.V₀)
  end

  return A

end

function assemble_forcing(FE_space::FEMProblem, param::ParametricSpecifics)
  #=MODIFY
  =#

  F = assemble_vector(∫(FE_space.ϕᵥ * param.f) * FE_space.dΩ + ∫(FE_space.ϕᵥ * param.h) * FE_space.dΓ, FE_space.V₀)

  return F

end

function assemble_H1_norm_matrix(FE_space::FEMProblem)
  #=MODIFY
  =#

  Xᵘ = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀) +
  assemble_matrix(∫(FE_space.ϕᵥ * FE_space.ϕᵤ) * FE_space.dΩ, FE_space.V, FE_space.V₀)

  return Xᵘ

end

function FE_solve(FE_space::FESpacePoisson, probl::ProblemSpecifics, param::ParametricSpecifics)
  #=MODIFY
  =#

  _, Gₕ = get_lifting_operator(FE_space, param)

  a(u, v) = ∫(∇(v) ⋅ (param.α * ∇(u))) * FE_space.dΩ
  b(v) = ∫(v * param.f) * FE_space.dΩ + ∫(v * param.h) * FE_space.dΓ
  operator = AffineFEOperator(a, b, FE_space.V, FE_space.V₀)

  if probl.solver === "lu"
    uₕ_field = solve(LinearFESolver(LUSolver()), operator)
  else
    uₕ_field = solve(LinearFESolver(), operator)
  end

  uₕ = get_free_dof_values(uₕ_field) - Gₕ

  return uₕ, Gₕ

end

function get_lifting_operator(FE_space::FESpacePoisson, param::ParametricSpecifics)

  gₕ = interpolate_everywhere(param.g, FE_space.V)
  Gₕ = get_free_dof_values(gₕ)

  gₕ, Gₕ

end
