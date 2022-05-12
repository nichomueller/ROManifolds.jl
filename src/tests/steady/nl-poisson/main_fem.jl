include("config_fem.jl")
include("../../../FEM/FEM.jl")

function run_FEM_0()

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x) = 1
  g(x) = [0., 1.]
  h(x) = 1

  FE_space = get_FE_space(problem_info, model, g)
  FE_space0 = get_FE_space(problem_info, model)

  function run_parametric_FEM(μ::Array)

    α(x) = μ[1] * (1 + x.^2)
    dα(dx, x) = 2 * μ[1] * x * dx

    parametric_info = ParametricSpecifics(μ, model, α, f, g, h)
    RHS = assemble_forcing(FE_space, parametric_info)
    LHS = assemble_stiffness(FE_space, problem_info, parametric_info) # FE_space must take into account the Dirichlet BCs
    Xᵘ = assemble_H1_norm_matrix(FE_space0)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return RHS, LHS, Xᵘ, parametric_solution

  end

  return run_parametric_FEM

end

function run_FEM_1()

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x) = 1
  g(x) = [0., 1.]
  h(x) = 1

  FE_space = get_FE_space(problem_info, model, g)
  FE_space0 = get_FE_space(problem_info, model)

  function run_parametric_FEM(μ::Array)

    α(x) = 1 + exp(-(x - μ[1])^2 / μ[1])
    dα(x) = (-exp(-(x - μ[1])^2 / μ[1]) .* 2*(x-μ[1])/μ[1])*dx

    parametric_info = ParametricSpecifics(μ, model, α, f, g, h)
    RHS = assemble_forcing(FE_space, parametric_info)
    LHS = assemble_stiffness(FE_space, problem_info, parametric_info)
    Xᵘ = assemble_H1_norm_matrix(FE_space0)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return RHS, LHS, Xᵘ, parametric_solution

  end

  return run_parametric_FEM

end

function run_FEM_2()

  model = DiscreteModelFromFile(paths.mesh_path)
  h(x) = 1

  FE_space = get_FE_space(problem_info, model)

  function run_parametric_FEM(μ::Array)

    @assert length(μ) === 5 "μ must be a 5x1 vector"
    α(x) = 1 + exp(-(x - μ[1])^2 / μ[1])
    dα(dx, x) = (-exp(-(x - μ[1])^2 / μ[1]) .* 2*(x-μ[1])/μ[1])*dx
    f(x) = sin(μ[4] * x[1]) + sin(μ[4] * x[2])
    g(x) = [0., sin(μ[5] * x[1]) + sin(μ[5] * x[2])]
    parametric_info = ParametricSpecifics(μ, model, α, f, g, h)

    RHS = assemble_forcing(FE_space, parametric_info)
    LHS = assemble_stiffness(FE_space, problem_info, parametric_info)
    Xᵘ = assemble_H1_norm_matrix(FE_space)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info, subtract_Ddata = false)

    end

    return RHS, LHS, Xᵘ, parametric_solution

  end

  return run_parametric_FEM

end

const μ = generate_parameter(ranges[1, :], ranges[2, :], nₛ)
save_CSV(μ, joinpath(problem_info.paths.FEM_snap_path, "μ.csv"))

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

save_CSV(lazy_solution_info[1][3], joinpath(paths.FEM_structures_path, "Xᵘ.csv"))
if case === 0
  save_CSV(lazy_solution_info[1][1], joinpath(paths.FEM_structures_path, "F.csv"))
  save_CSV(lazy_solution_info[1][2], joinpath(paths.FEM_structures_path, "A.csv"))
elseif case === 1
  save_CSV(lazy_solution_info[1][1], joinpath(paths.FEM_structures_path, "F.csv"))
end

uₕ = lazy_solution_info[1][4]()[1]
gₕ = lazy_solution_info[1][4]()[2]

if nₛ > 1
  uₕ = hcat(uₕ, zeros(size(uₕ)[1], nₛ - 1))
  gₕ = hcat(gₕ, zeros(size(gₕ)[1], nₛ - 1))
end
for i in 2:nₛ
  uₕ[:, i] = lazy_solution_info[i][4]()[1]
  gₕ[:, i] = lazy_solution_info[i][4]()[2]
end
save_CSV(uₕ, joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))
save_CSV(gₕ, joinpath(problem_info.paths.FEM_snap_path, "gₕ.csv"))
