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


function assemble_stiffness(FESpace, Param, FEMInfo)
    #=MODIFY
    =#

    if FEMInfo.problem_nonlinearities["A"] === false
        A = assemble_matrix(∫(∇(FESpace.ϕᵥ) ⋅ ∇(FESpace.ϕᵤ)) * FESpace.dΩ, FESpace.V, FESpace.V₀)
        save_variable(A, "A", "csv", joinpath(FEMInfo.paths.FEM_structures_path, "A.csv"))
    else
        A = assemble_matrix(∫(∇(FESpace.ϕᵥ) ⋅ (Param.α * ∇(FESpace.ϕᵤ))) * FESpace.dΩ, FESpace.V, FESpace.V₀)
    end

    A

end


function assemble_forcing(FESpace, Param, FEMInfo)
    #=MODIFY
    =#

    F = assemble_vector(∫( FESpace.ϕᵥ * Param.f ) * FESpace.dΩ + ∫( FESpace.ϕᵥ * Param.h ) * FESpace.dΓ, FESpace.V₀)
    if FEMInfo.problem_nonlinearities["f"] === false && FEMInfo.problem_nonlinearities["h"] === false
        save_variable(F, "F", "csv", joinpath(FEMInfo.paths.FEM_structures_path, "F.csv"))
    end

    F

end


function assemble_H1_norm_matrix(FESpace, Param, FEMInfo)
    #=MODIFY
    =#

    if FEMInfo.problem_nonlinearities["A"] === false
        Xᵘ = assemble_matrix(∫(∇(FESpace.ϕᵥ) ⋅ (Param.α * ∇(FESpace.ϕᵤ))) * FESpace.dΩ, FESpace.V, FESpace.V₀) + \
        assemble_matrix(∫(FESpace.ϕᵥ * (Param.α * FESpace.ϕᵤ)) * FESpace.dΩ, FESpace.V, FESpace.V₀)

    else # in this case, we do not consider the variable viscosity in the definition of the H1 norm matrix
        Xᵘ = assemble_matrix(∫(∇(FESpace.ϕᵥ) ⋅ ∇(FESpace.ϕᵤ)) * FESpace.dΩ, FESpace.V, FESpace.V₀) + \
        assemble_matrix(∫(FESpace.ϕᵥ * FESpace.ϕᵤ) * FESpace.dΩ, FESpace.V, FESpace.V₀)

    end

    save_variable(Xᵘ, "Xᵘ", "csv", joinpath(FEMInfo.paths.FEM_structures_path, "Xᵘ.csv"))
    Xᵘ

end


function FESpace_poisson(FEMInfo, Param)
    #=MODIFY
    =#

    Tₕ = Triangulation(Param.model)
    degree = 2 .* FEMInfo.order
    Qₕ = CellQuadrature(Tₕ, degree)

    ref_FE = ReferenceFE(lagrangian, Float64, FEMInfo.order)
    V₀ = TestFESpace(Param.model, ref_FE; conformity = :H1, dirichlet_tags = FEMInfo.dirichlet_tags)
    if FEMInfo.problem_nonlinearities["g"] === false
        V = TrialFESpace(V₀, Param.g)
    else
        V = V₀
    end
    ϕᵥ = get_fe_basis(V₀)
    ϕᵤ = get_trial_fe_basis(V)
    σₖ = get_cell_dof_ids(V₀)
    Nₕ = length(get_free_dof_ids(V))

    Ω = Triangulation(Param.model)
    dΩ = Measure(Ω, degree)
    Γ = BoundaryTriangulation(Param.model, tags = FEMInfo.neumann_tags)
    dΓ = Measure(Γ, degree)

    return FE_specifics(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, dΩ, dΓ)

end


function solve_poisson(FEMInfo, Param, FESpace, LHS, RHS)
    #=MODIFY
    =#

    a(u, v) = ∫(∇(v) ⋅ (Param.α * ∇(u))) * FESpace.dΩ
    f(v) = ∫( v * Param.f ) * FESpace.dΩ + ∫( v * Param.h ) * FESpace.dΓ

    if FEMInfo.problem_nonlinearities["g"] === false

        operator = AffineFEOperator(a, f, FESpace.V, FESpace.V₀)

        if FEMInfo.solver === "lu"
            uₕ_field = solve(LinearFESolver(LUSolver()), operator)
        else
            uₕ_field = solve(LinearFESolver(), operator)
        end
        uₕ = get_free_dof_values(uₕ_field)

    else

        uₕ = solve_poisson_lifting(FEMInfo, Param, FESpace, LHS, RHS)

    end

    uₕ

end


function solve_poisson_lifting(FEMInfo, Param, FESpace, LHS, RHS)
    #=MODIFY
    =#

    gₕ = interpolate_dirichlet(Param.g, FESpace.V₀)
    rₕ = ([integrate(∇(FESpace.ϕᵥ) ⋅ ∇(gₕ), FESpace.Qₕ)], [FESpace.σₖ])
    assembler = SparseMatrixAssembler(FESpace.V, FESpace.V₀)
    Rₕ = allocate_vector(assembler, rₕ)
    assemble_vector!(Rₕ, assembler, rₕ)

    if FEMInfo.lin_solver === "lu"
        lu_factors = lu(LHS)
        uₕ = lu_factors.U \ (lu_factors.L \ (RHS .- Rₕ))
    else
        uₕ = LHS \ (RHS .- Rₕ)
    end

    uₕ

end


#= function get_dof_values(uₕ, FESpace)
    #=MODIFY
    =#

    uₕ_free_dof_values = get_free_dof_values(uₕ)
    uₕ_dirichlet_dof_values = get_dirichlet_dof_values(FESpace.V)
    m = Broadcasting(PosNegReindex(uₕ_free_dof_values, uₕ_dirichlet_dof_values))
    lazy_map(m, FESpace.σₖ)[:]

end =#
