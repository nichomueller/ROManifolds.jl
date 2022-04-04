include("S-GRB_Poisson.jl")
include("RB_Poisson_unsteady.jl")

function get_RB_LHS_blocks(ROM_info, RB_variables::RB_problem; param = nothing)
  #=MODIFY
  =#

  initialize_RB_system(RB_variables::RB_problem)

  if ROM_info.problem_nonlinearities["A"]
    Aₙ_μ = assemble_stiffness(FE_space, ROM_info, param)
    (_, Aₙ_μ_affine) = MDEIM_online(Aₙ_μ, RB_variables.Aₙ_affine, RB_variables.Aₙ_idx)
    MAₙ = RB_variables.Mₙ + 2 / 3 * ROM_info.dt * Aₙ_μ_affine
  else
    MAₙ = RB_variables.Mₙ + 2 / 3 * ROM_info.dt * RB_variables.Aₙ
  end

  Φₜᵘ_1 = RB_variables.Φₜᵘ[2:end, :]' * RB_variables.Φₜᵘ[1:end - 1, :]
  Φₜᵘ_2 = RB_variables.Φₜᵘ[3:end, :]' * RB_variables.Φₜᵘ[1:end - 2, :]

  for i_s = 1:RB_variables.nₛᵘ
    for i_t = 1:RB_variables.nₜᵘ

      i_st = index_mapping(i_s, i_t)

      for j_s = 1:RB_variables.nₛᵘ
        for j_t = 1:RB_variables.nₜᵘ

          j_st = index_mapping(j_s, j_t)
          RB_variables.LHSₙ[1][i_st, j_st] = MAₙ[i_s, j_s] * (i_t == j_t) \
          - 4 / 3 * RB_variables.Mₙ[i_s, j_s] * Φₜᵘ_1[i_t, j_t] \
          + 1 / 3 * RB_variables.Mₙ[i_s, j_s] * Φₜᵘ_2[i_t, j_t]

        end
      end
    end
  end

  if ROM_info.save_offline_structures && !ROM_info.problem_nonlinearities["A"]
    save_variable(RB_variables.LHSₙ[1], "LHSₙ1", "csv", joinpath(ROM_info.paths.ROM_structures_path, "LHSₙ1"))
  end


end

function get_RB_RHS_blocks(ROM_info, RB_variables::RB_problem; param = nothing)
  #=MODIFY
  =#

  initialize_RB_system(RB_variables::RB_problem)

  if ROM_info.problem_nonlinearities["f"] || ROM_info.problem_nonlinearities["h"]
    Fₙ_μ = assemble_forcing(FE_space, param)
    (_, Fₙ_μ_affine) = DEIM_online(Fₙ_μ, RB_variables.Fₙ_affine, RB_variables.Fₙ_idx)
    Fₙ = 2 / 3 * ROM_info.dt * Fₙ_μ_affine
  else
    Fₙ = 2 / 3 * ROM_info.dt * RB_variables.Fₙ
  end

  for i_s = 1:RB_variables.nₛᵘ
    for i_t = 1:RB_variables.nₜᵘ

      RB_variables.RHSₙ[1][index_mapping(i_s, i_t)] = Fₙ[i_s] * Φₜᵘ[:, i_t]

    end
  end

  if ROM_info.save_offline_structures && !ROM_info.problem_nonlinearities["f"] && !ROM_info.problem_nonlinearities["h"]
    save_variable(RB_variables.RHSₙ[1], "RHSₙ1", "csv", joinpath(ROM_info.paths.ROM_structures_path, "RHSₙ1"))
  end


end
