include("../utils/general.jl")

function FE_space(probl::problem_specifics, param::parametric_specifics)
  #=MODIFY
  =#

  Tₕ = Triangulation(param.model)
  degree = 2 .* probl.order
  Qₕ = CellQuadrature(Tₕ, degree)

  ref_FE = ReferenceFE(lagrangian, Float64, probl.order)
  V₀ = TestFESpace(param.model, ref_FE; conformity = :H1, dirichlet_tags = probl.dirichlet_tags)
  V = TrialFESpace(V₀, param.g)

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  σₖ = get_cell_dof_ids(V₀)
  Nₕ = length(get_free_dof_ids(V))

  Ω = Triangulation(param.model)
  dΩ = Measure(Ω, degree)
  Γ = BoundaryTriangulation(param.model, tags = probl.neumann_tags)
  dΓ = Measure(Γ, degree)

  return FESpacePoisson(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, dΩ, dΓ)

end

function FE_space_0(probl::problem_specifics, model::UnstructuredDiscreteModel)
  #=MODIFY
  =#

  Tₕ = Triangulation(model)
  degree = 2 .* probl.order
  Qₕ = CellQuadrature(Tₕ, degree)

  ref_FE = ReferenceFE(lagrangian, Float64, probl.order)
  V₀ = TestFESpace(model, ref_FE; conformity = :H1, dirichlet_tags = probl.dirichlet_tags)
  V = TrialFESpace(V₀, (x->0))

  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  σₖ = get_cell_dof_ids(V₀)
  Nₕ = length(get_free_dof_ids(V))

  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γ = BoundaryTriangulation(model, tags = probl.neumann_tags)
  dΓ = Measure(Γ, degree)

  return FESpacePoisson(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, dΩ, dΓ)

end

function assemble_stiffness(FE_space::FEMProblem, probl::problem_specifics, param::parametric_specifics)
    #=MODIFY
    =#

    if probl.problem_nonlinearities["A"] === false
        A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀)
    else
        A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (param.α * ∇(FE_space.ϕᵤ))) * FE_space.dΩ, FE_space.V, FE_space.V₀)
    end

    return A

end

function assemble_forcing(FE_space::FEMProblem, param::parametric_specifics)
    #=MODIFY
    =#

    F = assemble_vector(∫( FE_space.ϕᵥ * param.f ) * FE_space.dΩ + ∫( FE_space.ϕᵥ * param.h ) * FE_space.dΓ, FE_space.V₀)

    return F

end

function assemble_H1_norm_matrix(FE_space::FEMProblem, probl::problem_specifics)
    #=MODIFY
    =#

    Xᵘ = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀) + \
    assemble_matrix(∫(FE_space.ϕᵥ * FE_space.ϕᵤ) * FE_space.dΩ, FE_space.V, FE_space.V₀)
    save_variable(Xᵘ, "Xᵘ", "csv", joinpath(probl.paths.FEM_structures_path, "Xᵘ.csv"))

    return Xᵘ

end

function FE_solve(FE_space::FESpacePoisson, probl::problem_specifics, param::parametric_specifics)
    #=MODIFY
    =#

    a(u, v) = ∫(∇(v) ⋅ (param.α * ∇(u))) * FE_space.dΩ
    f(v) = ∫( v * param.f ) * FE_space.dΩ + ∫( v * param.h ) * FE_space.dΓ
    operator = AffineFEOperator(a, f, FE_space.V, FE_space.V₀)

    if probl.solver === "lu"
      uₕ_field = solve(LinearFESolver(LUSolver()), operator)
    else
      uₕ_field = solve(LinearFESolver(), operator)
    end

    uₕ = get_free_dof_values(uₕ_field)
    return uₕ

end

function FE_solve_lifting(FE_space::FESpacePoisson, probl::problem_specifics, param::parametric_specifics)
    #=MODIFY
    =#

    gₕ = interpolate_everywhere(param.g, FE_space.V₀)

    a(u, v) = ∫(∇(v) ⋅ (param.α * ∇(u))) * FE_space.dΩ
    f(v) = ∫( v * param.f ) * FE_space.dΩ + ∫( v * param.h ) * FE_space.dΓ - a(gₕ, v)
    operator = AffineFEOperator(a, f, FE_space.V, FE_space.V₀)

    if probl.solver === "lu"
      uₕ_field = solve(LinearFESolver(LUSolver()), operator)
    else
      uₕ_field = solve(LinearFESolver(), operator)
    end

    uₕ = get_free_dof_values(uₕ_field) + get_free_dof_values(gₕ)
    return uₕ

end
