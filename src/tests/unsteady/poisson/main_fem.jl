include("config_fem.jl")

function run_FEM_0()

  function run_parametric_FEM(μ::Array)

    parametric_info = get_parametric_specifics(problem_ntuple,problem_info,μ)

    A = assemble_stiffness(FE_space, problem_info, parametric_info)(0.0)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, M, H, F, A

  end

  return run_parametric_FEM

end

function run_FEM_1()

  function run_parametric_FEM(μ::Array)

    parametric_info = get_parametric_specifics(problem_ntuple,problem_info,μ)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    F = assemble_forcing(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, M, H, F

  end

  return run_parametric_FEM

end

function run_FEM_2()

  function run_parametric_FEM(μ::Array)

    parametric_info = get_parametric_specifics(problem_ntuple,problem_info,μ)
    M = assemble_mass(FE_space, problem_info, parametric_info)(0.0)
    H = assemble_neumann_datum(FE_space, problem_info, parametric_info)(0.0)
    Xᵘ₀ = assemble_H1_norm_matrix_nobcs(FE_space₀)

    function parametric_solution()

      return FE_solve(FE_space, problem_info, parametric_info)

    end

    return parametric_solution, Xᵘ₀, M, H

  end

  return run_parametric_FEM

end

FEM_time₀ = @elapsed begin

  const μ = generate_parameter(ranges[1, :], ranges[2, :], nₛ)

  if case == 0
    FEM = run_FEM_0()
  elseif case == 1
    FEM = run_FEM_1()
  else case == 2
    FEM = run_FEM_2()
  end

  lazy_solution_info = lazy_map(FEM, μ)

  Xᵘ₀ = lazy_solution_info[1][2]
  M = lazy_solution_info[1][3]
  H = lazy_solution_info[1][4]
  if case == 0
    A = lazy_solution_info[1][6]
    F = lazy_solution_info[1][5]
  elseif case == 1
    F = lazy_solution_info[1][5]
  end

end

FEM_time₁ = @elapsed begin
  #gₕ = lazy_solution_info[1][1]()[2]
  Nₕ = FE_space.Nₛᵘ
  Nₜ = convert(Int64, T/δt)
  uₕ = zeros(Nₕ,nₛ*Nₜ)
  for i in 1:nₛ
    @info "Collecting solution number $i"
    uₕ[:,(i-1)*Nₜ+1:i*Nₜ] = lazy_solution_info[i][1]()[1]
    #gₕ[:, (i-1)*Nₜ+1:i*Nₜ] = lazy_solution_info[i][1]()[2]
  end
end

FEM_time = FEM_time₀+FEM_time₁/nₛ

save_CSV(μ, joinpath(problem_info.paths.FEM_snap_path, "μ.csv"))
save_CSV(Xᵘ₀, joinpath(paths.FEM_structures_path, "Xᵘ₀.csv"))
save_CSV(H, joinpath(paths.FEM_structures_path, "H.csv"))
save_CSV(M, joinpath(paths.FEM_structures_path, "M.csv"))
save_CSV([FEM_time], joinpath(paths.FEM_structures_path, "FEM_time.csv"))

if case == 0
  save_CSV(A, joinpath(paths.FEM_structures_path, "A.csv"))
  save_CSV(F, joinpath(paths.FEM_structures_path, "F.csv"))
elseif case == 1
  save_CSV(F, joinpath(paths.FEM_structures_path, "F.csv"))
end

save_CSV(uₕ, joinpath(problem_info.paths.FEM_snap_path, "uₕ.csv"))
#save_CSV(gₕ, joinpath(problem_info.paths.FEM_snap_path, "gₕ.csv"))
