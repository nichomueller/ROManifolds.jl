

function assemble_stiffness(FE_space, nonlinear_A = false)
    #=MODIFY
    =#

    if nonlinear_A === false
        a(u, v) = ∫(∇(v) ⋅ ∇(u)) * dΩ
        A = assemble_matrix(a(FE_space.ϕᵤ, FE_space.ϕᵥ), FE_space.V, FE_space.V₀)
    else
        a(u, v) = ∫(∇(v) ⋅ (parametric_info.α() * ∇(u))) * dΩ
        A = assemble_matrix(a(FE_space.ϕᵤ, FE_space.ϕᵥ), FE_space.V, FE_space.V₀)
    end

    (out) -> (a; A)

end
(out) -> (a; A)


function assemble_forcing(FE_space, parametric_info)
    #=MODIFY
    =#

    f(v) = ∫( v * parametric_info.f() ) * dΩ + ∫( v * parametric_info.h() ) * dΓ
    F = assemble_vector(f(v), FE_space.V, FE_space.V₀)
    (out) -> (f; F)

end
(out) -> (f; F)


function FE_space_poisson(problem_info, parametric_info)
    #=MODIFY
    =#

    Tₕ = Triangulation(parametric_info.model)
    degree = 2 .* problem_info.order 
    Qₕ = CellQuadrature(Tₕ, degree)

    ref_FE = ReferenceFE(lagrangian, Float64, problem_info.order)
    V₀ = TestFESpace(parametric_info.model, ref_FE; conformity = :H1, dirichlet_tags = problem_info.dirichlet_tags)
    if problem_info.problem_nonlinearities["g"] === false
        V = TrialFESpace(V₀, parametric_info.g())
    else
        V = V₀
    end
    ϕᵥ = get_fe_basis(V₀)
    ϕᵤ = get_trial_fe_basis(V)

    Ω = Triangulation(parametric_info.model)
    dΩ = Measure(Ω, degree)
    Γ = BoundaryTriangulation(parametric_info.model, tags = problem_info.neumanntags)
    dΓ = Measure(Γ, degree)
    
    (out) -> (Qₕ; V₀; V; ϕᵥ; ϕᵤ; dΩ; dΓ)

end
(out) -> (Qₕ; V₀; V; ϕᵥ; ϕᵤ; dΩ; dΓ)


function solve_poisson(problem_info, parametric_info, FE_space, LHS, RHS, nₛ)
    #=MODIFY
    =#

    uₕ = zeros(length(get_free_dof_ids(FE_space.V)) + length(get_dirichlet_dof_ids(FE_space.V)), nₛ)
    uₕ = solve_poisson!(problem_info, parametric_info, FE_space, LHS, RHS)
end


function solve_poisson(problem_info, parametric_info, FE_space, LHS, RHS)
    #=MODIFY
    =#

    if problem_info.problem_nonlinearities["g"] === false

        operator = AffineFEOperator(RHS.a(), LHS.f(), FE_space.V₀, FE_space.V)

        if problem_info.lin_solver === "LU"
            uₕ = solve(LinearFESolver(LUSolver()), operator) 
        else
            uₕ = solve(LinearFESolver(), operator) 
        end

    else

        uₕ = solve_poisson_lifting(problem_info, parametric_info, FE_space, LHS, RHS)

    end

    uₕ

end


function solve_poisson_lifting(problem_info, parametric_info, FE_space, LHS, RHS)
    #=MODIFY
    =#

    gₕ = interpolate_dirichlet(parametric_info.g(), FE_space.V₀)
    Rₕ = integrate(∇(FE_space.ϕᵥ) ⋅ ∇(gₕ), FE_space.Qₕ) 
   
    if problem_info.lin_solver === "lu"
        (L, U) = lu(RHS.A)
        uₕ = U \ (L \ (LHS.F .- Rₕ))
    else
        uₕ = RHS.A \ (LHS.F .- Rₕ)
    end

    uₕ

end


