include("config_fom.jl")
include("../../../FEM/FOM.jl")

all_paths = FEM_ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim)
problem_info = problem_specifics(problem_name, problem_type, all_paths, problem_type, problem_name, approx_type, problem_dim, problem_nonlinearities, number_coupled_blocks, order, dirichlet_tags, neumann_tags, solver)

ranges = Dict("μᵒ" => [0., 1.], "μᴬ" => [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]], "μᶠ" => [0., 1.1], "μᵍ" => [0., 1.], "μʰ" => [0., 1.])
params = generate_parameters(problem_nonlinearities, nₛ, ranges)
parametric_info = compute_parametric_info(problem_nonlinearities, params, 1)
FE_space = FE_space_poisson(problem_data, parametric_info)
uₕ = zeros(length(get_free_dof_ids(FE_space.V)) + length(get_dirichlet_dof_ids(FE_space.V)), nₛ)

if problem_nonlinearities["Ω"] === false

    if problem_nonlinearities["f"] === false && problem_nonlinearities["h"] === false
        RHS = assemble_forcing(FE_space, parametric_info)
    end
    if problem_nonlinearities["A"] === false 
        LHS = assemble_stiffness(FE_space)
    end

    for i_nₛ = 1:nₛ
        if i_nₛ > 1
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
        end
        if problem_nonlinearities["f"] === true || problem_nonlinearities["h"] === true
            RHS = assemble_forcing(FE_space, parametric_info)
        end
        if problem_nonlinearities["A"] === true 
            LHS = assemble_stiffness(FE_space)
        end
        push!(uₕ, solve_poisson(problem_info, parametric_info, FE_space, LHS, RHS))
    end

else

    for i_nₛ = 1:nₛ
        if i_nₛ > 1
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
            FE_space = FE_space_poisson(problem_data, parametric_info)
        end
        if problem_nonlinearities["f"] === true || problem_nonlinearities["h"] === true
            RHS = assemble_forcing(FE_space, parametric_info)
        end
        if problem_nonlinearities["A"] === true 
            LHS = assemble_stiffness(FE_space)
        end
        push!(uₕ, solve_poisson(problem_info, parametric_info, FE_space, LHS, RHS))
    end

end

