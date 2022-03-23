

function assemble_nonlin_stiffness(μ)
    #=MODIFY
    =#

    a(u, v) = ∫(μ * ( ∇(v) ⋅ ∇(u) )) * dΩ
    a

end


function generate_FOM_poisson(FOM_info, mesh_path)
    #=MODIFY
    =#

    model = DiscreteModelFromFile(mesh_path)
 
    ref_FE = ReferenceFE(lagrangian, Float64, FOM_info.order)
    V0 = TestFESpace(model, ref_FE; conformity = :H1, dirichlet_tags = FOM_info.dirichlet_tags)
    Vg = TrialFESpace(V0, FOM_info.g)

    degree = 2 .* FOM_info.order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    Γ = BoundaryTriangulation(model, tags = FOM_info.neumanntags)
    dΓ = Measure(Γ, degree)

    a(u, v) = ∫( ∇(v) ⋅ ∇(u) ) * dΩ
    f(v) = ∫( v * FOM_info.f ) * dΩ + ∫( v * FOM_info.h ) * dΓ
    operator = AffineFEOperator(a, f, Vg, V0)

    if FOM_info.lin_solver !== nothing
        solver = LinearFESolver(FOM_info.lin_solver)
    else
        solver = LinearFESolver()
    end

    uh = solve(solver, operator)

    dv = get_fe_basis(V0)
    du = get_trial_fe_basis(Vg)
    A, F = assemble_matrix_and_vector(a(du, dv), f(v), Vg, V0)
    (A, F, uh)

end


