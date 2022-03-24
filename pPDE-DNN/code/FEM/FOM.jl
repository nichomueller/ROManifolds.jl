struct FE_space_info
    Qₕ
    V₀
    V
    ϕᵥ
    ϕᵤ
    dΩ
    dΓ
end


function assemble_stiffness(FE_space, α)
    #=MODIFY
    =#

    a(u, v) = ∫(∇(v) ⋅ (α * ∇(u))) * dΩ
    A = assemble_matrix(a(FE_space.du, FE_space.dv), FE_space.Vg, FE_space.V0)
    return a, A

end


function assemble_forcing(FE_space)
    #=MODIFY
    =#

    f(v) = ∫( v * FOM_info.f ) * dΩ + ∫( v * FOM_info.h ) * dΓ
    F = assemble_vector(f(v), FE_space.Vg, FE_space.V0)
    return f, F

end


function FOM_poisson(FOM_info, mesh_path)
    #=MODIFY
    =#

    model = DiscreteModelFromFile(mesh_path)
    Tₕ = Triangulation(model)
    degree = 2 .* FOM_info.order 
    Qₕ = CellQuadrature(Tₕ, degree)

    ref_FE = ReferenceFE(lagrangian, Float64, FOM_info.order)
    V₀ = TestFESpace(model, ref_FE; conformity = :H1, dirichlet_tags = FOM_info.dirichlet_tags)
    V = TrialFESpace(V₀, FOM_info.g)
    ϕᵥ = get_fe_basis(V₀)
    ϕᵤ = get_trial_fe_basis(V)

    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    Γ = BoundaryTriangulation(model, tags = FOM_info.neumanntags)
    dΓ = Measure(Γ, degree)
    
    FE_space = FE_space_info(Qₕ, V₀, V, ϕᵥ, ϕᵤ, dΩ, dΓ)

    a, A = assemble_stiffness(FE_space, μ)
    f, F = assemble_forcing(FE_space)
    operator = AffineFEOperator(a, f, V₀, V)

    if FOM_info.lin_solver === "LU"
        solver = LinearFESolver(LUSolver())
    else
        solver = LinearFESolver()
    end

    uh = solve(solver, operator)

    dv = get_fe_basis(V₀)
    du = get_trial_fe_basis(V₀)
    A, F = assemble_matrix_and_vector(a(du, dv), f(v), V₀, V)
    (A, F, uh)

end


function FOM_poisson_lifting(FE_space, g, μ)
    #=MODIFY
    =#

    gₕ = interpolate_dirichlet(g, FE_space.V₀)
    Rₕ = integrate((∇(FE_space.ϕᵥ) ⋅ (μ * ∇(gₕ))), FE_space.Qₕ) 
    _, A = assemble_stiffness(FE_space, μ)
    _, F = assemble_forcing(FE_space)

    uh = A \ (F .- Rₕ)

    return (A, F, uh)

end



