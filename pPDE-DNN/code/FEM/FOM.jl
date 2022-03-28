

struct FE_specifics
    Qₕ
    V₀
    V
    ϕᵥ
    ϕᵤ
    σₖ
    Nₕ
    dΩ
    dΓ
end


function assemble_stiffness(FE_space, parametric_info, problem_info)
    #=MODIFY
    =#

    if problem_info.problem_nonlinearities["A"] === false
        A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ ∇(FE_space.ϕᵤ)) * FE_space.dΩ, FE_space.V, FE_space.V₀)
        save_variable(A, "A", "csv", joinpath(problem_info.paths.FEM_structures_path, "A.csv"))
    else
        A = assemble_matrix(∫(∇(FE_space.ϕᵥ) ⋅ (parametric_info.α * ∇(FE_space.ϕᵤ))) * FE_space.dΩ, FE_space.V, FE_space.V₀)
    end

    A

end


function assemble_forcing(FE_space, parametric_info)
    #=MODIFY
    =#

    assemble_vector(∫( FE_space.ϕᵥ * parametric_info.f ) * FE_space.dΩ + ∫( FE_space.ϕᵥ * parametric_info.h ) * FE_space.dΓ, FE_space.V₀) 

end


function FE_space_poisson(problem_info, parametric_info)
    #=MODIFY
    =#

    Tₕ = Triangulation(parametric_info.model)
    degree = 2 .* problem_info.order 
    Qₕ = CellQuadrature(Tₕ, degree)

    ref_FE = ReferenceFE(lagrangian, Float64, problem_info.order)
    V₀ = TestFESpace(parametric_info.model, ref_FE; conformity = :H1, dirichlet_tags = problem_info.dirichlet_tags)
    if problem_info.problem_nonlinearities["g"] === false
        V = TrialFESpace(V₀, parametric_info.g)
    else
        V = V₀
    end
    ϕᵥ = get_fe_basis(V₀)
    ϕᵤ = get_trial_fe_basis(V)
    σₖ = get_cell_dof_ids(V₀)
    Nₕ = length(get_free_dof_ids(V)) #+ length(get_dirichlet_dof_ids(V))

    Ω = Triangulation(parametric_info.model)
    dΩ = Measure(Ω, degree)
    Γ = BoundaryTriangulation(parametric_info.model, tags = problem_info.neumann_tags)
    dΓ = Measure(Γ, degree)
    
    return FE_specifics(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, dΩ, dΓ)
    #() -> (Qₕ; V₀; V; ϕᵥ; ϕᵤ; dΩ; dΓ)

end


function solve_poisson(problem_info, parametric_info, FE_space, LHS, RHS)
    #=MODIFY
    =#

    a(u, v) = ∫(∇(v) ⋅ (parametric_info.α * ∇(u))) * FE_space.dΩ
    f(v) = ∫( v * parametric_info.f ) * FE_space.dΩ + ∫( v * parametric_info.h ) * FE_space.dΓ

    if problem_info.problem_nonlinearities["g"] === false

        operator = AffineFEOperator(a, f, FE_space.V, FE_space.V₀)

        if problem_info.solver === "lu"
            uₕ_field = solve(LinearFESolver(LUSolver()), operator) 
        else
            uₕ_field = solve(LinearFESolver(), operator) 
        end
        uₕ = get_free_dof_values(uₕ_field)

    else

        uₕ = solve_poisson_lifting(problem_info, parametric_info, FE_space, LHS, RHS)

    end

    uₕ

end


function solve_poisson_lifting(problem_info, parametric_info, FE_space, LHS, RHS)
    #=MODIFY
    =#

    gₕ = interpolate_dirichlet(parametric_info.g, FE_space.V₀)
    rₕ = ([integrate(∇(FE_space.ϕᵥ) ⋅ ∇(gₕ), FE_space.Qₕ)], [FE_space.σₖ])
    assembler = SparseMatrixAssembler(FE_space.V, FE_space.V₀)
    Rₕ = allocate_vector(assembler, rₕ)
    assemble_vector!(Rₕ, assembler, rₕ)
   
    if problem_info.lin_solver === "lu"
        lu_factors = lu(LHS[2])
        uₕ = lu_factors.U \ (lu_factors.L \ (RHS.F .- Rₕ))
    else
        uₕ = LHS[2] \ (RHS.F .- Rₕ)
    end

    uₕ

end


#= function get_dof_values(uₕ, FE_space)
    #=MODIFY
    =#

    uₕ_free_dof_values = get_free_dof_values(uₕ)
    uₕ_dirichlet_dof_values = get_dirichlet_dof_values(FE_space.V)
    m = Broadcasting(PosNegReindex(uₕ_free_dof_values, uₕ_dirichlet_dof_values))
    lazy_map(m, FE_space.σₖ)[:]

end =#
