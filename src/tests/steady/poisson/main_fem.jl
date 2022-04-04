include("config_fem.jl")
include("../../../FEM/FEM.jl")

function run_FEM()

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x) = 1
  g(x) = 1
  h(x) = 1

  FE_space = FE_space_0(problem_info, model)

  function run_parametric_FEM(μ::Array)

    @assert length(μ) === 3 "μ must be a 3x1 vector"
    α(x) = sum(μ)

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

function run_FEM_A()

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x) = 1
  g(x) = 1
  h(x) = 1

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

function run_FEM_A_f_g()

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

  α(x) = 1
  f(x) = 1
  g(x) = 1
  h(x) = 1

  ref_info = reference_info(1, 3, 100)

  function run_parametric_FEM(μ::Array)

    @assert length(μ) === 1 "μ must be a 1x1 vector"

    model = generate_cartesian_model(ref_info, stretching, μ)
    parametric_info = parametric_specifics(μ, model, α, f, g, h)
    FE_space = FE_space(problem_info, parametric_info)

    RHS = assemble_forcing(FE_space, problem_info, parametric_info)
    LHS = assemble_stiffness(FE_space, problem_info, parametric_info)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return RHS, LHS, parametric_solution

  end

  return run_parametric_FEM

end

const μ = generate_parameter(ranges[1, :], ranges[2, :], nₛ)
save_variable(μ, "μ", "csv", joinpath(problem_info.paths.FEM_snap_path, "μ.csv"))

if case === 0
  FEM = run_FEM()
elseif case === 1
  FEM = run_FEM_A()
elseif case === 2
  FEM = run_FEM_A_f_g()
else
  FEM = run_FEM_Omega()
end

lazy_solution_info = lazy_map(FEM, μ)

if case === 0
  save_variable(lazy_solution_info[1][1], "F", "csv", joinpath(paths.FEM_structures_path, "F.csv"))
  save_variable(lazy_solution_info[1][2], "A", "csv", joinpath(paths.FEM_structures_path, "A.csv"))
elseif case === 1
  save_variable(lazy_solution_info[1][1], "F", "csv", joinpath(paths.FEM_structures_path, "F.csv"))
end

uₕ = lazy_solution_info[1][3]()
if nₛ > 1
  uₕ = hcat(uₕ, zeros(size(uₕ)[1], nₛ - 1))
end
for i in 1:nₛ
  uₕ[:, i] = lazy_solution_info[i][3]()
end
save_variable(uₕ, "uₕ", "csv", joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))
