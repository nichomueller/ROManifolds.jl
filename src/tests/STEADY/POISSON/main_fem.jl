include("config_fem.jl")
include("../../../FEM/FEM.jl")
#include("../../../ROM/RB_utils.jl")


function run_FEM()

  problem_name = "poisson"
  problem_type = "steady"
  problem_dim = 3
  problem_nonlinearities = Dict("Ω" => false, "A" => true, "f" => false, "g" => false, "h" => false)
  mesh_name = "model.json"
  root = "/home/user1/git_repos/Mabla.jl"
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)

  nₛ = 100
  order = 1
  dirichlet_tags = ["sides"]
  neumann_tags = ["circle", "triangle", "square"]
  solver = "lu"

  problem_info = problem_specifics(order, dirichlet_tags, neumann_tags, solver, nₛ, paths, problem_nonlinearities)

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x) = 1
  g(x) = 1
  h(x) = 1

  Tₕ = Triangulation(model)
  degree = 2 .* order
  Qₕ = CellQuadrature(Tₕ, degree)
  ref_FE = ReferenceFE(lagrangian, Float64, order)
  V₀ = TestFESpace(model, ref_FE; conformity = :H1, dirichlet_tags = dirichlet_tags)
  V = TrialFESpace(V₀, g)
  ϕᵥ = get_fe_basis(V₀)
  ϕᵤ = get_trial_fe_basis(V)
  σₖ = get_cell_dof_ids(V₀)
  Nₕ = length(get_free_dof_ids(V))
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γ = BoundaryTriangulation(model, tags = neumann_tags)
  dΓ = Measure(Γ, degree)

  FE_space = FESpacePoisson(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, dΩ, dΓ)

  function parametric_part(μ::Array)

    α(x) = μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])

    parametric_info = parametric_specifics(μ, model, α, f, g, h)
    RHS = assemble_forcing(FE_space, problem_info, parametric_info)
    LHS = assemble_stiffness(FE_space, problem_info, parametric_info)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return RHS, LHS, parametric_solution

  end

  return parametric_part

end

FEM = run_FEM()
ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]]
μ = generate_parameter(ranges[1, :], ranges[2, :])
(RHS, LHS, parametric_solution) = FEM(μ)
uₕ = parametric_solution()



#= paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)
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

save_variable(uₕ, "uₕ", "csv", joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))=#




#= S = assemble_stiffness(FE_space, compute_parametric_info(problem_nonlinearities, params, 1), problem_info)[:]
for i_nₛ = 2:nₛ-1
    parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
    S = hcat(S, assemble_stiffness(FE_space, parametric_info, problem_info)[:])
end
(MDEIM_mat, MDEIM_idx) = DEIM_offline(S, 1e-3)
LHS10 = assemble_stiffness(FE_space, compute_parametric_info(problem_nonlinearities, params, 10), problem_info)
(MDEIM_coeffs, mat_affine) = MDEIM_online(LHS10, MDEIM_mat, MDEIM_idx)
err_MDEIM = norm(LHS10 - mat_affine)/norm(LHS10) =#
