include("config_fem.jl")
include("../../../FEM/FEM.jl")

α(x) = μ[1] * (1 + x.^2)
dα(dx, x) = 2 * μ[1] * x * dx

function run_FEM_0()

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x) = 1
  g(x) = 1
  h(x) = 1

  FE_space = get_FE_space(problem_info, model, g)
  FE_space0 = get_FE_space(problem_info, model)

  function run_parametric_FEM(μ::Array)

    @assert length(μ) === 3 "μ must be a 3x1 vector"
    α(x) = sum(μ)

    parametric_info = ParametricSpecifics(μ, model, α, f, g, h)
    F = assemble_forcing(FE_space, problem_info, parametric_info)
    A = assemble_stiffness(FE_space, problem_info, parametric_info)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space0)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    parametric_solution, Xᵘ₀, F, A

  end

  return run_parametric_FEM

end

function run_FEM_1()

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x) = 1
  g(x) = 1
  h(x) = 1

  FE_space = get_FE_space(problem_info, model, g)
  FE_space0 = get_FE_space(problem_info, model)

  function run_parametric_FEM(μ::Array)

    @assert length(μ) === 3 "μ must be a 3x1 vector"
    α(x) = 1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])

    parametric_info = ParametricSpecifics(μ, model, α, f, g, h)
    F = assemble_forcing(FE_space, problem_info, parametric_info)
    A = assemble_stiffness(FE_space, problem_info, parametric_info)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space0)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    parametric_solution, Xᵘ₀, F, A

  end

  return run_parametric_FEM

end

function run_FEM_2()

  model = DiscreteModelFromFile(paths.mesh_path)
  h(x) = 1

  FE_space = get_FE_space(problem_info, model)

  function run_parametric_FEM(μ::Array)

    @assert length(μ) === 5 "μ must be a 5x1 vector"
    α(x) = 1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) / μ[3])
    f(x) = sin(μ[4] * x[1]) + sin(μ[4] * x[2])
    g(x) = sin(μ[5] * x[1]) + sin(μ[5] * x[2])
    parametric_info = ParametricSpecifics(μ, model, α, f, g, h)

    F = assemble_forcing(FE_space, problem_info, parametric_info)
    A = assemble_stiffness(FE_space, problem_info, parametric_info)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info, subtract_Ddata = false)

    end

    parametric_solution, Xᵘ₀, F, A

  end

  return run_parametric_FEM

end

FEM_time₀ = @elapsed begin

  const μ = generate_parameter(ranges[1, :], ranges[2, :], nₛ)

  if case === 0
    FEM = run_FEM_0()
  elseif case === 1
    FEM = run_FEM_1()
  elseif case === 2
    FEM = run_FEM_2()
  else
    FEM = run_FEM_3()
  end

  lazy_solution_info = lazy_map(FEM, μ)

  Xᵘ₀ = lazy_solution_info[1][2]
  H = lazy_solution_info[1][3][2]
  if case === 0
    A = lazy_solution_info[1][4]
    F = lazy_solution_info[1][3][1]
  elseif case === 1
    F = lazy_solution_info[1][3][1]
  end

end

FEM_time₁ = @elapsed begin

  uₕ = lazy_solution_info[1][1]()[1]
  #gₕ = lazy_solution_info[1][1]()[2]

  if nₛ > 1
    uₕ = hcat(uₕ, zeros(size(uₕ)[1], nₛ - 1))
    #gₕ = hcat(gₕ, zeros(size(gₕ)[1], nₛ - 1))
  end
  for i in 2:nₛ
    uₕ[:, i] = lazy_solution_info[i][1]()[1]
    #gₕ[:, i] = lazy_solution_info[i][1]()[2]
  end
  save_CSV(uₕ, joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))
  #save_CSV(gₕ, joinpath(problem_info.paths.FEM_snap_path, "gₕ.csv"))

end

FEM_time = FEM_time₀+FEM_time₁/nₛ

save_CSV(μ, joinpath(problem_info.paths.FEM_snap_path, "μ.csv"))
save_CSV(Xᵘ₀, joinpath(paths.FEM_structures_path, "Xᵘ₀.csv"))
save_CSV(H, joinpath(paths.FEM_structures_path, "H.csv"))
save_CSV([FEM_time], joinpath(paths.FEM_structures_path, "FEM_time.csv"))

if case === 0
  save_CSV(A, joinpath(paths.FEM_structures_path, "A.csv"))
  save_CSV(F, joinpath(paths.FEM_structures_path, "F.csv"))
elseif case === 1
  save_CSV(F, joinpath(paths.FEM_structures_path, "F.csv"))
end

save_CSV(uₕ, joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))
#save_CSV(gₕ, joinpath(problem_info.paths.FEM_snap_path, "gₕ.csv"))
