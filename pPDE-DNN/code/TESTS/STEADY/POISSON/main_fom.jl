include("config_fom.jl")
include("../../../FEM/FOM.jl")

all_paths = FEM_ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)
problem_info = problem_specifics(problem_name, problem_type, all_paths, approx_type, problem_dim, problem_nonlinearities, number_coupled_blocks, order, dirichlet_tags, neumann_tags, solver)

ranges = Dict("μᵒ" => [0., 1.], "μᴬ" => [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]], "μᶠ" => [0., 1.1], "μᵍ" => [0., 1.], "μʰ" => [0., 1.])
(μᵒ, μᴬ, μᶠ, μᵍ, μʰ) = generate_parameters(problem_nonlinearities, nₛ, ranges)
params = param_info(μᵒ, μᴬ, μᶠ, μᵍ, μʰ)

#= parametric_info = compute_parametric_info(problem_nonlinearities, params, 1)
FE_space = FE_space_poisson(problem_info, parametric_info)
RHS = assemble_forcing(FE_space, parametric_info)   
LHS = assemble_stiffness(FE_space, parametric_info, problem_info)
uₕ = Vector{Float64}(undef, FE_space.Nₕ)  =#
uₕ = Any[]

if problem_nonlinearities["Ω"] === false 

    parametric_info = compute_parametric_info(problem_nonlinearities, params, 1)
    FE_space = FE_space_poisson(problem_info, parametric_info)

    for i_nₛ = 1:nₛ
        @info "Computing snapshot $i_nₛ"

        if i_nₛ > 1
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
        end

        if problem_nonlinearities["f"] === true || problem_nonlinearities["h"] === true
            RHS = assemble_forcing(FE_space, parametric_info)
        end
        if problem_nonlinearities["A"] === true
            LHS = assemble_stiffness(FE_space, parametric_info, problem_info)
        end
        if problem_nonlinearities["A"] === false
            LHS .*= parametric_info.α(Point(0., 0.))
        end

        push!(uₕ, solve_poisson(problem_info, parametric_info, FE_space, LHS, RHS))

    end

else

    for i_nₛ = 1:nₛ
        @info "Computing snapshot $i_nₛ"
        
        parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
        FE_space = FE_space_poisson(problem_info, parametric_info)
        RHS = assemble_forcing(FE_space, parametric_info)
        LHS = assemble_stiffness(FE_space, parametric_info, problem_specifics)
       
        
        push!(uₕ, solve_poisson(problem_info, parametric_info, FE_space, LHS, RHS))

    end

end

save_variable(uₕ, "uₕ", "csv", joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))