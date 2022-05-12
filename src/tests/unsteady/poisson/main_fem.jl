include("config_fem.jl")
include("../../../FEM/FEM.jl")

αₛ(x) = 1
αₜ(t, μ) = sum(μ) * (2 + sin(2π * t))
mₛ(x) = 1
mₜ(t::Real) = 1
m(x, t::Real) = mₛ(x)*mₜ(t)
m(t::Real) = x -> m(x, t)
fₛ(x) = 1
fₜ(t::Real) = sin(π * t)
gₛ(x) = 0
gₜ(t::Real) = 0
g(x, t::Real) = gₛ(x)*gₜ(t)
g(t::Real) = x -> g(x, t)
hₛ(x) = 0
hₜ(t::Real) = 0
h(x, t::Real) = hₛ(x)*hₜ(t)
h(t::Real) = x -> h(x, t)
u₀(x) = 0

function run_FEM_0()

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x, t::Real) = fₛ(x)*fₜ(t)
  f(t::Real) = x -> f(x, t)
  FE_space = get_FE_space(problem_info, model, g)

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = αₛ(x)*αₜ(t, μ)
    α(t::Real) = x -> α(x, t)
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, g, hₛ, hₜ, h, u₀)

    A = assemble_stiffness(FE_space, problem_info, parametric_info)(0.0)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, M, A

  end

  return run_parametric_FEM

end

function run_FEM_1()

  model = DiscreteModelFromFile(paths.mesh_path)
  f(x, t::Real) = fₛ(x)*fₜ(t)
  f(t::Real) = x -> f(x, t)
  FE_space = get_FE_space(problem_info, model, g)

  function run_parametric_FEM(μ::Array)

    αₛ(x) = 0
    αₜ(t::Real) = 0
    α(x, t::Real) = (1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) * sin(t) / μ[3]))
    α(t::Real) = x -> α(x, t)
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, g, hₛ, hₜ, h, u₀)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, M

  end

  return run_parametric_FEM

end

function run_FEM_2()

  model = DiscreteModelFromFile(paths.mesh_path)
  FE_space = get_FE_space(problem_info, model)

  function run_parametric_FEM(μ::Array)

    αₛ(x) = 0
    αₜ(t::Real) = 0
    α(x, t::Real) = (1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) * sin(t) / μ[3]))
    α(t::Real) = x -> α(x, t)
    f(x, t::Real) = sin(π*t*x*(μ[4]+μ[5]))
    f(t::Real) = x -> f(x, t)
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, g, hₛ, hₜ, h, u₀)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, M

  end

  return run_parametric_FEM

end

const μ = generate_parameter(ranges[1, :], ranges[2, :], nₛ)
save_CSV(μ, joinpath(problem_info.paths.FEM_snap_path, "μ.csv"))

if case === 0
  FEM = run_FEM_0()
elseif case === 1
  FEM = run_FEM_1()
else case === 2
  FEM = run_FEM_2()
end

lazy_solution_info = lazy_map(FEM, μ)

save_CSV(lazy_solution_info[1][2], joinpath(paths.FEM_structures_path, "Xᵘ₀.csv"))
save_CSV(lazy_solution_info[1][3][2], joinpath(paths.FEM_structures_path, "H.csv"))
save_CSV(lazy_solution_info[1][4], joinpath(paths.FEM_structures_path, "M.csv"))

if case === 0
  save_CSV(lazy_solution_info[1][5], joinpath(paths.FEM_structures_path, "A.csv"))
  save_CSV(lazy_solution_info[1][3][1], joinpath(paths.FEM_structures_path, "F.csv"))
elseif case === 1
  save_CSV(lazy_solution_info[1][3][1], joinpath(paths.FEM_structures_path, "F.csv"))
end

@info "Collecting solution number 1"
uₕ = lazy_solution_info[1][1]()[1]
#gₕ = lazy_solution_info[1][1]()[2]

Nₜ = convert(Int64, T / δt)
if nₛ > 1
  uₕ = hcat(uₕ, zeros(size(uₕ)[1], (nₛ - 1)*Nₜ))
  #gₕ = hcat(gₕ, zeros(size(gₕ)[1], (nₛ - 1)*Nₜ))
end
for i in 2:nₛ
  @info "Collecting solution number $i"
  uₕ[:, (i-1)*Nₜ+1:i*Nₜ] = lazy_solution_info[i][1]()[1]
  #gₕ[:, (i-1)*Nₜ+1:i*Nₜ] = lazy_solution_info[i][1]()[2]
end
save_CSV(uₕ, joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))
#save_CSV(gₕ, joinpath(problem_info.paths.FEM_snap_path, "gₕ.csv"))
