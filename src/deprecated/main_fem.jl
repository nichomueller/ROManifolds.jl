include("config_fem.jl")
include("../../../FEM/FEM.jl")
#include("../../../ROM/RB_utils.jl")

function run_FEM_A()

  # problem_nonlinearities = Dict("Ω" => false, "A" => true, "f" => true, "g" => true, "h" => false)

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x) = 1
  g(x) = 1
  h(x) = 1

  #= Tₕ = Triangulation(model)
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

  FE_space = FESpacePoisson(Qₕ, V₀, V, ϕᵥ, ϕᵤ, σₖ, Nₕ, dΩ, dΓ) =#

  FE_space = FE_space_0(problem_info, model)

  function run_parametric_FEM(μ::Array)

    @assert length(μ) === 3 "μ must be a 3x1 vector"
    α(x) = μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])

    parametric_info = parametric_specifics(μ, model, α, f, g, h)
    RHS = assemble_forcing(FE_space, problem_info, parametric_info)
    LHS = assemble_stiffness(FE_space, problem_info, parametric_info)

    function parametric_solution()

      return FE_solve_lifting(FE_space, problem_info, parametric_info)

    end

    return RHS, LHS, parametric_solution

  end

  return run_parametric_FEM

end

const FEM = run_FEM_A()
const nₛ = 100
const ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]]
const μ = generate_parameter(ranges[1, :], ranges[2, :], nₛ)
save_variable(μ, "μ", "csv", joinpath(problem_info.paths.FEM_snap_path, "μ.csv"))
const lazy_solution_info = lazy_map(FEM, μ)

save_variable(lazy_solution_info[1][1], "F", "csv", joinpath(paths.FEM_structures_path, "F.csv"))
uₕ = lazy_solution_info[1][3]()
if nₛ > 1
  uₕ = hcat(uₕ, zeros(size(uₕ)[1], nₛ - 1))
end
for i in 1:nₛ
  uₕ[:, i] = lazy_solution_info[i][3]()
end
save_variable(uₕ, "uₕ", "csv", joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))

function run_FEM_A_f_g()

  # problem_nonlinearities = Dict("Ω" => false, "A" => true, "f" => true, "g" => true, "h" => false)

  model = DiscreteModelFromFile(paths.mesh_path)
  h(x) = 1

  FE_space = FE_space_0(problem_info, model)

  function run_parametric_FEM(μ::Array)

    @assert length(μ) === 5 "μ must be a 5x1 vector"
    α(x) = μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])
    f(x) = sin(μ[4] * x[1]) + sin(μ[4] * x[2])
    g(x) = sin(μ[5] * x[1]) + sin(μ[5] * x[2])
    parametric_info = parametric_specifics(μ, model, α, f, g, h)

    RHS = assemble_forcing(FE_space, problem_info, parametric_info)
    LHS = assemble_stiffness(FE_space, problem_info, parametric_info)

    function parametric_solution()

      return FE_solve_lifting(FE_space, problem_info, parametric_info)

    end

    return RHS, LHS, parametric_solution

  end

  return run_parametric_FEM

end

function run_FEM_Omega()

  # problem_nonlinearities = Dict("Ω" => true, "A" => false, "f" => false, "g" => false, "h" => false)
  # mesh_name = "model.json"
  # dirichlet_tags = "boundary"
  # neumann_tags = []

  α(x) = 1
  f(x) = 1
  g(x) = 1
  h(x) = 0

  ref_info = reference_info(1, 3, 100)

  function run_parametric_FEM(μ::Array)

    @assert length(μ) === 1 "μ must be a 1x1 vector"

    model = generate_cartesian_model(ref_info, stretching, μ)
    parametric_info = parametric_specifics(μ, model, α, f, g, h)
    FE_space = FE_space(problem_info, parametric_info)

    RHS = assemble_forcing(FE_space, problem_info, parametric_info)
    LHS = assemble_stiffness(FE_space, problem_info, parametric_info)

    function parametric_solution()

      return FE_solve_lifting(FE_space, problem_info, parametric_info)

    end

    return RHS, LHS, parametric_solution

  end

  return run_parametric_FEM

end
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
