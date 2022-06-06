include("S-GRB_Poisson.jl")
include("RB_Poisson_unsteady.jl")

function get_RB_LHS_blocks(RBInfo, RBVars::RB_problem, Param; FESpace = nothing)
  #=MODIFY
  to be used when BDF2 is employed
  =#

  initialize_RB_system(RBVars::RB_problem)

  if RBInfo.problem_nonlinearities["A"]
    Aₙ_μ = assemble_stiffness(FESpace, RBInfo, Param)
    (_, Aₙ_μ_affine) = MDEIM_online(Aₙ_μ, RBVars.Aₙ_affine, RBVars.Aₙ_idx)
    MAₙ = RBVars.Mₙ + 2 / 3 * RBInfo.dt * Aₙ_μ_affine
  else
    MAₙ = RBVars.Mₙ + 2 / 3 * RBInfo.dt * RBVars.Aₙ * Param.μ
  end

  Φₜᵘ_1 = RBVars.Φₜᵘ[2:end, :]' * RBVars.Φₜᵘ[1:end - 1, :]
  Φₜᵘ_2 = RBVars.Φₜᵘ[3:end, :]' * RBVars.Φₜᵘ[1:end - 2, :]

  for i_s = 1:RBVars.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t)

      for j_s = 1:RBVars.nₛᵘ
        for j_t = 1:RBVars.nₜᵘ

          j_st = index_mapping(j_s, j_t)
          RBVars.LHSₙ[1][i_st, j_st] = MAₙ[i_s, j_s] * (i_t == j_t) \
          - 4 / 3 * RBVars.Mₙ[i_s, j_s] * Φₜᵘ_1[i_t, j_t] \
          + 1 / 3 * RBVars.Mₙ[i_s, j_s] * Φₜᵘ_2[i_t, j_t]

        end
      end
    end
  end

  if RBInfo.save_offline_structures && !RBInfo.problem_nonlinearities["A"]
    save_variable(RBVars.LHSₙ[1], "LHSₙ1", "csv", joinpath(RBInfo.paths.ROM_structures_path, "LHSₙ1"))
  end


end

function get_RB_RHS_blocks(RBInfo, RBVars::RB_problem, Param; FESpace = nothing)


  initialize_RB_system(RBVars::RB_problem)

  if RBInfo.problem_nonlinearities["f"] || RBInfo.problem_nonlinearities["h"]
    Fₙ_μ = assemble_forcing(FESpace, Param)
    (_, Fₙ_μ_affine) = DEIM_online(Fₙ_μ, RBVars.Fₙ_affine, RBVars.Fₙ_idx)
    Fₙ = 2 / 3 * RBInfo.dt * Fₙ_μ_affine
  else
    Fₙ = 2 / 3 * RBInfo.dt * RBVars.Fₙ
  end

  for i_s = 1:RBVars.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      RBVars.RHSₙ[1][index_mapping(i_s, i_t)] = Fₙ[i_s] * Φₜᵘ[:, i_t]

    end
  end

  if RBInfo.save_offline_structures && !RBInfo.problem_nonlinearities["f"] && !RBInfo.problem_nonlinearities["h"]
    save_variable(RBVars.RHSₙ[1], "RHSₙ1", "csv", joinpath(RBInfo.paths.ROM_structures_path, "RHSₙ1"))
  end


end
