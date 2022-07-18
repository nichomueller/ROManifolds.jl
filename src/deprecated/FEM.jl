struct FE_Info
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


function assemble_stiffness(FEMSpace, Param, FEMInfo)
    #=MODIFY
    =#

    if FEMInfo.problem_nonlinearities["A"] === false
        A = assemble_matrix(∫(∇(FEMSpace.ϕᵥ) ⋅ ∇(FEMSpace.ϕᵤ)) * FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)
        save_variable(A, "A", "csv", joinpath(FEMInfo.paths.FEM_structures_path, "A.csv"))
    else
        A = assemble_matrix(∫(∇(FEMSpace.ϕᵥ) ⋅ (Param.α * ∇(FEMSpace.ϕᵤ))) * FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)
    end

    A

end


function assemble_forcing(FEMSpace, Param, FEMInfo)
    #=MODIFY
    =#

    F = assemble_vector(∫( FEMSpace.ϕᵥ * Param.f ) * FEMSpace.dΩ + ∫( FEMSpace.ϕᵥ * Param.h ) * FEMSpace.dΓ, FEMSpace.V₀)
    if FEMInfo.problem_nonlinearities["f"] === false && FEMInfo.problem_nonlinearities["h"] === false
        save_variable(F, "F", "csv", joinpath(FEMInfo.paths.FEM_structures_path, "F.csv"))
    end

    F

end


function assemble_H1_norm_matrix(FEMSpace, Param, FEMInfo)
    #=MODIFY
    =#

    if FEMInfo.problem_nonlinearities["A"] === false
        Xᵘ = assemble_matrix(∫(∇(FEMSpace.ϕᵥ) ⋅ (Param.α * ∇(FEMSpace.ϕᵤ))) * FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀) + \
        assemble_matrix(∫(FEMSpace.ϕᵥ * (Param.α * FEMSpace.ϕᵤ)) * FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)

    else # in this case, we do not consider the variable viscosity in the definition of the H1 norm matrix
        Xᵘ = assemble_matrix(∫(∇(FEMSpace.ϕᵥ) ⋅ ∇(FEMSpace.ϕᵤ)) * FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀) + \
        assemble_matrix(∫(FEMSpace.ϕᵥ * FEMSpace.ϕᵤ) * FEMSpace.dΩ, FEMSpace.V, FEMSpace.V₀)

    end

    save_variable(Xᵘ, "Xᵘ", "csv", joinpath(FEMInfo.paths.FEM_structures_path, "Xᵘ.csv"))
    Xᵘ

end


function FEMSpace_poisson(FEMInfo, Param)
    #=MODIFY
    =#

    Tₕ = Triangulation(Param.model)
    degree = 2 .* FEMInfo.order
    Qₕ = CellQuadrature(Tₕ, degree)

    ref_FE = Gridap.ReferenceFE(lagrangian, Float, FEMInfo.order)
    V₀ = TestFEMSpace(Param.model, ref_FE; conformity = :H1, dirichlet_tags = FEMInfo.dirichlet_tags)
    if FEMInfo.problem_nonlinearities["g"] === false
        V = TrialFEMSpace(V₀, Param.g)
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

    return FE_Info(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, dΩ, dΓ)

end


function solve_poisson(FEMInfo, Param, FEMSpace, LHS, RHS)
    #=MODIFY
    =#

    a(u, v) = ∫(∇(v) ⋅ (Param.α * ∇(u))) * FEMSpace.dΩ
    f(v) = ∫( v * Param.f ) * FEMSpace.dΩ + ∫( v * Param.h ) * FEMSpace.dΓ

    if FEMInfo.problem_nonlinearities["g"] === false

        operator = AffineFEOperator(a, f, FEMSpace.V, FEMSpace.V₀)

        if FEMInfo.solver === "lu"
            uₕ_field = solve(LinearFESolver(LUSolver()), operator)
        else
            uₕ_field = solve(LinearFESolver(), operator)
        end
        uₕ = get_free_dof_values(uₕ_field)

    else

        uₕ = solve_poisson_lifting(FEMInfo, Param, FEMSpace, LHS, RHS)

    end

    uₕ

end


function solve_poisson_lifting(FEMInfo, Param, FEMSpace, LHS, RHS)
    #=MODIFY
    =#

    gₕ = interpolate_dirichlet(Param.g, FEMSpace.V₀)
    rₕ = ([integrate(∇(FEMSpace.ϕᵥ) ⋅ ∇(gₕ), FEMSpace.Qₕ)], [FEMSpace.σₖ])
    assembler = SparseMatrixAssembler(FEMSpace.V, FEMSpace.V₀)
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


#= function get_dof_values(uₕ, FEMSpace)
    #=MODIFY
    =#

    uₕ_free_dof_values = get_free_dof_values(uₕ)
    uₕ_dirichlet_dof_values = get_dirichlet_dof_values(FEMSpace.V)
    m = Broadcasting(PosNegReindex(uₕ_free_dof_values, uₕ_dirichlet_dof_values))
    lazy_map(m, FEMSpace.σₖ)[:]

end =#
