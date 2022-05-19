include("config_fem.jl")

αₛ(x) = 1
αₜ(t::Real, μ) = sum(μ) * (2 + sin(2π * t))
mₛ(x) = 1
mₜ(t::Real) = 1
m(x, t::Real) = mₛ(x)*mₜ(t)
m(t::Real) = x -> m(x, t)
fₛ(x) = VectorValue(0,0,0)
fₜ(t::Real) = 0
f(x, t::Real) = fₛ(x)*fₜ(t)
f(t::Real) = x -> f(x, t)
gʷ(x, t::Real) = 0
gʷ(t::Real) = x -> gʷ(x, t)
const x₀ = Point(0,0,0)
const R = 0.41784
gₛ(x) = 2 * (1 .- VectorValue((x[1]-x₀[1])^2,(x[2]-x₀[2])^2,(x[3]-x₀[3])^2) / R^2) / (pi*R^2)
gₜ(t::Real, μ) = 1-cos(2*pi*t/T)+μ[2]*sin(2*pi*μ[1]*t/T)
hₛ(x) = VectorValue(0,0,0)
hₜ(t::Real) = 0
h(x, t::Real) = hₛ(x)*hₜ(t)
h(t::Real) = x -> h(x, t)
u₀(x) = VectorValue(0,0,0)
p₀(x) = VectorValue(0,0,0)

function run_FEM_0()

  model = DiscreteModelFromFile(paths.mesh_path)

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = αₛ(x)*αₜ(t, μ)
    α(t::Real) = x -> α(x, t)
    g(x, t::Real) = gₛ(x)*gₜ(t, μ)
    g(t::Real) = x -> g(x, t)
    FE_space = get_FE_space(problem_info, model; [gʷ,g])
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, [gʷ,g], hₛ, hₜ, h, [u₀,p₀])

    A = assemble_stiffness(FE_space, problem_info, parametric_info)(0.0)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, H, M, A

  end

  return run_parametric_FEM

end

function run_FEM_1()

  model = DiscreteModelFromFile(paths.mesh_path)
  FE_space = get_FE_space(problem_info, model, g)

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = (1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) * sin(t) / μ[3]))
    α(t::Real) = x -> α(x, t)
    g(x, t::Real) = gₛ(x)*gₜ(t, μ[4:end])
    g(t::Real) = x -> g(x, t)
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, [gʷ,g], hₛ, hₜ, h, u₀)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, H, M

  end

  return run_parametric_FEM

end

function run_FEM_2()

  model = DiscreteModelFromFile(paths.mesh_path)
  FE_space = get_FE_space(problem_info, model)

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = αₛ(x)*αₜ(t, μ)
    α(t::Real) = x -> α(x, t)
    gₜ(t::Real, μ) = gₜ(t, μ[end-2:end-1])*μ[end]
    g1(x, t::Real) = gₛ(x)*gₜ(t, μ)
    g1(t::Real) = x -> g(x, t)
    g2(x, t::Real) = gₛ(x)*(1-gₜ(t, μ))
    g2(t::Real) = x -> g(x, t)
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, [gʷ,g1,g2], hₛ, hₜ, h, u₀)
    A = assemble_stiffness(FE_space, problem_info, parametric_info)(0.0)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, H, M, A

  end

  return run_parametric_FEM

end

function run_FEM_3()

  model = DiscreteModelFromFile(paths.mesh_path)
  FE_space = get_FE_space(problem_info, model)

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = (1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) * sin(t) / μ[3]))
    α(t::Real) = x -> α(x, t)
    gₜ(t::Real, μ) = gₜ(t, μ[end-2:end-1])*μ[end]
    g1(x, t::Real) = gₛ(x)*gₜ(t, μ)
    g1(t::Real) = x -> g(x, t)
    g2(x, t::Real) = gₛ(x)*(1-gₜ(t, μ))
    g2(t::Real) = x -> g(x, t)
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, [gʷ,g1,g2], hₛ, hₜ, h, u₀)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, H, M

  end

  return run_parametric_FEM

end

FEM_time₀ = @elapsed begin

  const μ = generate_parameter(ranges[1, :], ranges[2, :], nₛ)

  if case === 0
    FEM = run_FEM_0()
  elseif case === 1
    FEM = run_FEM_1()
  else case === 2
    FEM = run_FEM_2()
  end

  lazy_solution_info = lazy_map(FEM, μ)

  (Xᵘ₀, F, H, M) = lazy_solution_info[1][2:5]
  if case === 0 || case === 2
    A = lazy_solution_info[1][6]
  end

end

FEM_time₁ = @elapsed begin
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
end

FEM_time = FEM_time₀+FEM_time₁/nₛ

save_CSV(μ, joinpath(problem_info.paths.FEM_snap_path, "μ.csv"))
save_CSV(Xᵘ₀, joinpath(paths.FEM_structures_path, "Xᵘ₀.csv"))
save_CSV(F, joinpath(paths.FEM_structures_path, "F.csv"))
save_CSV(H, joinpath(paths.FEM_structures_path, "H.csv"))
save_CSV(M, joinpath(paths.FEM_structures_path, "M.csv"))
save_CSV([FEM_time], joinpath(paths.FEM_structures_path, "FEM_time.csv"))

if case === 0 || case === 2
  save_CSV(A, joinpath(paths.FEM_structures_path, "A.csv"))
end

save_CSV(uₕ, joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))
#save_CSV(gₕ, joinpath(problem_info.paths.FEM_snap_path, "gₕ.csv"))
