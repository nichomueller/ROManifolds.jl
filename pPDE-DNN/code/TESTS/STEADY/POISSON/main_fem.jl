Pkg.activate(".")

include("config_fem.jl")
include("../../../FEM/FEM.jl")
include("../../../ROM/RB_utils.jl")

paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)
problem_info = problem_specifics(problem_name, problem_type, paths, approx_type, problem_dim, problem_nonlinearities, number_coupled_blocks, order, dirichlet_tags, neumann_tags, solver, nₛ)

ranges = Dict("μᵒ" => [0., 1.], "μᴬ" => [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]], "μᶠ" => [0., 1.1], "μᵍ" => [0., 1.], "μʰ" => [0., 1.])
(μᵒ, μᴬ, μᶠ, μᵍ, μʰ) = generate_parameters(problem_nonlinearities, nₛ, ranges)
params = param_info(μᵒ, μᴬ, μᶠ, μᵍ, μʰ)
parametric_info = compute_parametric_info(problem_nonlinearities, params, 1)

FE_space = FE_space_poisson(problem_info, parametric_info)
RHS = assemble_forcing(FE_space, parametric_info, problem_info)   
LHS = assemble_stiffness(FE_space, parametric_info, problem_info)
#= uₕ = Vector{Float64}(undef, FE_space.Nₕ)  =#
uₕ = Any[]

if problem_nonlinearities["Ω"] === false 

    for i_nₛ = 1:nₛ
        @info "Computing snapshot $i_nₛ"

        if i_nₛ > 1
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
            if problem_nonlinearities["f"] === true || problem_nonlinearities["h"] === true
                RHS = assemble_forcing(FE_space, parametric_info, problem_info)
            end
            if problem_nonlinearities["A"] === true
                LHS = assemble_stiffness(FE_space, parametric_info, problem_info)
            end
        end

        if problem_nonlinearities["A"] === false
            LHS .*= parametric_info.α(Point(0., 0.))
        end

        push!(uₕ, solve_poisson(problem_info, parametric_info, FE_space, LHS, RHS))

    end

else

    for i_nₛ = 1:nₛ
        @info "Computing snapshot $i_nₛ"
        
        if i_nₛ > 1
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
            FE_space = FE_space_poisson(problem_info, parametric_info)
            RHS = assemble_forcing(FE_space, parametric_info, problem_info)
            LHS = assemble_stiffness(FE_space, parametric_info, problem_info)
        end
        
        push!(uₕ, solve_poisson(problem_info, parametric_info, FE_space, LHS, RHS))

    end

end

save_variable(uₕ, "uₕ", "csv", joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))

#= S = assemble_stiffness(FE_space, compute_parametric_info(problem_nonlinearities, params, 1), problem_info)[:]
for i_nₛ = 2:nₛ-1
    parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
    S = hcat(S, assemble_stiffness(FE_space, parametric_info, problem_info)[:])
end
(MDEIM_mat, MDEIM_idx) = DEIM_offline(S, 1e-3)
LHS10 = assemble_stiffness(FE_space, compute_parametric_info(problem_nonlinearities, params, 10), problem_info)
(MDEIM_coeffs, mat_affine) = MDEIM_online(LHS10, MDEIM_mat, MDEIM_idx)
err_MDEIM = norm(LHS10 - mat_affine)/norm(LHS10) =#