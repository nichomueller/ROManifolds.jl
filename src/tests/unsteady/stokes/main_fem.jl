include("config_fem.jl")

function run_FEM_0()

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = αₛ(x)*αₜ(t, μ)
    α(t::Real) = x -> α(x, t)
    gⁱⁿ(x, t::Real) = gₛ(x)*gₜ(t, μ)
    gⁱⁿ(t::Real) = x -> gⁱⁿ(x, t)
    function g(x,t::Real)
      return gʷ(x,t) + gⁱⁿ(x,t)
    end
    g(t::Real) = x -> g(x, t)
    FE_space = get_FESpace(problem_ntuple, problem_info, model, [gʷ,gⁱⁿ])
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, g, hₛ, hₜ, h, x₀)

    A = assemble_stiffness(FE_space, problem_info, parametric_info)(0.0)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    Bᵀ = assemble_primal_opᵀ(FE_space)
    B = assemble_primal_op(FE_space)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, H, M, Bᵀ, B, A

  end

  return run_parametric_FEM

end

function run_FEM_1()

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = αₛ(x)*αₜ(t, μ)
    α(t::Real) = x -> α(x, t)
    gⁱⁿ(x, t::Real) = gₛ(x)*gₜ(t, μ)
    gⁱⁿ(t::Real) = x -> gⁱⁿ(x, t)
    function g(x,t::Real)
      return gʷ(x,t) + gⁱⁿ(x,t)
    end
    g(t::Real) = x -> g(x, t)
    FE_space = get_FESpace(problem_ntuple, problem_info, model, [gʷ,gⁱⁿ])
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, g, hₛ, hₜ, h, x₀)

    A = assemble_stiffness(FE_space, problem_info, parametric_info)(0.0)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    Bᵀ = assemble_primal_opᵀ(FE_space)
    B = assemble_primal_op(FE_space)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, H, M, Bᵀ, B, A

  end

  return run_parametric_FEM

end

function run_FEM_2()

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = (1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) * sin(t) / μ[3]))
    α(t::Real) = x -> α(x, t)
    g(x, t::Real) = gₛ(x)*gₜ(t, μ[4:end])
    g(t::Real) = x -> g(x, t)
    FE_space = get_FESpace(problem_info, model, g)
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, [gʷ,g], hₛ, hₜ, h, u₀)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    Bᵀ = assemble_primal_opᵀ(FE_space)
    B = assemble_primal_op(FE_space)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, H, M, Bᵀ, B

  end

  return run_parametric_FEM

end

function run_FEM_3()

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = αₛ(x)*αₜ(t, μ)
    α(t::Real) = x -> α(x, t)
    gₜ(t::Real, μ) = gₜ(t, μ[end-2:end-1])*μ[end]
    g1(x, t::Real) = gₛ(x)*gₜ(t, μ)
    g1(t::Real) = x -> g1(x, t)
    g2(x, t::Real) = gₛ(x)*(1-gₜ(t, μ))
    g2(t::Real) = x -> g2(x, t)
    FE_space = get_FESpace(problem_info, model, g)
    parametric_info = ParametricSpecificsUnsteady(μ, model, αₛ, αₜ, α, mₛ, mₜ, m, fₛ, fₜ, f, gₛ, gₜ, [gʷ,g1,g2], hₛ, hₜ, h, u₀)
    A = assemble_stiffness(FE_space, problem_info, parametric_info)(0.0)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    Bᵀ = assemble_primal_opᵀ(FE_space)
    B = assemble_primal_op(FE_space)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, F, H, M, Bᵀ, B, A

  end

  return run_parametric_FEM

end

function run_FEM_4()

  function run_parametric_FEM(μ::Array)

    α(x, t::Real) = (1 + μ[3] + 1 / μ[3] * exp(-((x[1] - μ[1])^2 + (x[2] - μ[2])^2) * sin(t) / μ[3]))
    α(t::Real) = x -> α(x, t)
    gₜ(t::Real, μ) = gₜ(t, μ[end-2:end-1])*μ[end]
    g1(x, t::Real) = gₛ(x)*gₜ(t, μ)
    g1(t::Real) = x -> g(x, t)
    g2(x, t::Real) = gₛ(x)*(1-gₜ(t, μ))
    g2(t::Real) = x -> g(x, t)
    FE_space = get_FESpace(problem_info, model, g)
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
  elseif case === 2
    FEM = run_FEM_2()
  elseif case === 3
    FEM = run_FEM_3()
  elseif case === 4
    FEM = run_FEM_4()
  end

  lazy_solution_info = lazy_map(FEM, μ)

  (Xᵘ₀, F, H, M, B, Bᵀ) = lazy_solution_info[1][2:7]
  if case === 0 || case === 2
    A = lazy_solution_info[1][8]
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
save_CSV(B, joinpath(paths.FEM_structures_path, "B.csv"))
save_CSV(Bᵀ, joinpath(paths.FEM_structures_path, "Bᵀ.csv"))
save_CSV([FEM_time], joinpath(paths.FEM_structures_path, "FEM_time.csv"))

if case === 0 || case === 2
  save_CSV(A, joinpath(paths.FEM_structures_path, "A.csv"))
end

save_CSV(uₕ, joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))
#save_CSV(gₕ, joinpath(problem_info.paths.FEM_snap_path, "gₕ.csv"))
