function get_snapshot_matrix(RBInfo, RBVars::RBProblem)

  @info "Importing the snapshot matrix, number of snapshots considered: $n_snap"

  name = "uₕ"
  Sᵘ = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, name * ".csv"), DataFrame))[:, 1:(RBInfo.nₛ*RBInfo.Nₜ)]
  RBVars.Sᵘ = Sᵘ
  RBVars.Nᵤˢ = size(Sᵘ)[1]

  @info "Dimension of snapshot matrix: $(size(Sᵘ)); (Nᵤˢ, Nₜ, nₛ) = ($RBVars.Nᵤˢ, $RBInfo.Nₜ, $RBInfo.nₛ)"

end

function PODs_time(RBInfo, RBVars::RBProblem)


  @info "Performing the temporal POD for field u, using a tolerance of $RBInfo.ϵₜ"

  if RBInfo.time_reduction_technique === "ST-HOSVD"
    Sᵘₜ = zeros(RBInfo.Nₜ, RBVars.nᵤˢ * RBInfo.nₛ)
    for i in 1:RBInfo.nₛ
      Sᵘₜ[:, RBVars.nᵤˢ*i:RBVars.nᵤˢ*(i+1)] = \
      transpose(Sᵘ[:, RBInfo.Nₜ*i:RBInfo.Nₜ*(i+1)])
    end
  else
    Sᵘₜ = zeros(RBInfo.Nₜ, RBVars.Nᵤˢ * RBInfo.nₛ)
    for i in 1:RBInfo.nₛ
      Sᵘₜ[:, RBVars.Nᵤˢ*i:RBVars.Nᵤˢ*(i+1)] = \
      transpose(Sᵘ[:, RBInfo.Nₜ*i:RBInfo.Nₜ*(i+1)])
    end
  end

  Φₜᵘ, _ = POD(Sᵘ, RBInfo.ϵₜ)
  RBVars.Φₜᵘ = Φₜᵘ
  RBVars.nₜᵘ = size(Φₛᵘ)[2]

end

function build_reduced_basis(RBInfo, RBVars::RBProblem)


  @info "Building the space-time reduced basis, using tolerances of $(RBInfo.ϵₛ, RBInfo.ϵₜ)"

  RB_building_time = @elapsed begin
    PODs_space(RBInfo, RBVars)
    PODs_time(RBInfo, RBVars)
  end

  RBVars.nᵘ = nₛᵘ * nₜᵘ
  RBVars.offline_time += RB_building_time

  if RBInfo.save_offline_structures
    save_CSV(Φₛᵘ, joinpath(RBInfo.paths.basis_path, "Φₛᵘ.csv"))
    save_CSV(Φₜᵘ, joinpath(RBInfo.paths.basis_path, "Φₜᵘ.csv"))
  end

end

function import_reduced_basis(RBInfo, RBVars::RBProblem)


  @info "Importing the reduced basis"

  RBVars.Φₛᵘ = load_CSV(joinpath(RBInfo.paths.basis_path, "Φₛᵘ.csv"))
  RBVars.Φₜᵘ = load_CSV(joinpath(RBInfo.paths.basis_path, "Φₜᵘ.csv"))
  (RBVars.Nₛᵘ, RBVars.nₛᵘ) = size(RBVars.Φₛᵘ)
  RBVars.nₜᵘ = size(RBVars.Φₜᵘ)[2]
  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ

end

function check_reduced_affine_components(RBInfo, RBVars::RBProblem)


  operators = []

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
    @info "Importing reduced mass matrix"
    RBVars.Mₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Mₙ.csv"))
  else
    @info "Failed to import the reduced mass matrix: must build it"
    push!(operators, "M")
  end

  push!(operators, check_reduced_affine_components(RBInfo, RBVars.S))

  operators

end

function index_mapping(i::T, j::T) where {T<:Int64}


  return int(i * nₜᵘ + j)

end

function get_generalized_coordinates(RBInfo, RBVars::RBProblem, snaps=nothing)


  if !check_norm_matrix(RBVars)
    get_norm_matrix(RBInfo, RBVars)
  end

  if snaps === nothing || maximum(snaps) > RBInfo.nₛ
    snaps = 1:RBInfo.nₛ
  end

  û = zeros(RBVars.nᵘ, length(snaps))
  Φₛᵘ_normed = RBVars.Xᵘ * RBVars.Φₛᵘ

  for i_nₛ = snaps
    @info "Assembling generalized coordinate relative to snapshot $(i_nₛ)"
    for i_s = 1:RBVars.nₛᵘ
      for i_t = 1:RBVars.nₛᵗ
        Π_ij = kron(Φₛᵘ_normed[:, i_s], RBVars.Φₜᵘ[:, i_t])
        S_ij = RBVars.Sᵘ[:, (i_nₛ-1)*RBInfo.Nₜ+1:i_nₛ*RBInfo.Nₜ]
        û[index_mapping(i_s, i_t), i_nₛ] = Π_ij' * S_ij
      end
    end
  end

  RBVars.û = û

  if RBInfo.save_offline_structures
    save_CSV(û, joinpath(RBInfo.paths.gen_coords_path, "û.csv"))
  end

end

function check_affine_blocks(RBInfo, RBVars::RBProblem)


  if RBInfo.import_offline_structures === false
    return ["A", "F"]
  end

  operators = []
  initialize_RB_system(RBVars)

  for i = 1:1

    LHSₙi = "LHSₙ" * string(i) * ".csv"
    RHSₙi = "RHSₙ" * string(i) * ".csv"

    if isfile(joinpath(RBInfo.paths.ROM_structures_path, LHSₙi * ".csv"))
      @info "Importing block number $i of the reduced affine LHS"
      push!(RBVars.LHSₙ, load_CSV(joinpath(RBInfo.paths.ROM_structures_path, LHSₙi)))
      RBVars.nₛᵘ = size(RBVars.LHSₙ[i])[1]
    else
      @info "Failed to import the block number $i of the reduced affine LHS: must build it"
      push!(operators, "A")
    end

    if isfile(joinpath(RBInfo.paths.ROM_structures_path, RHSₙi * ".csv"))
      @info "Importing block number $i of the reduced affine RHS"
      push!(RBVars.RHSₙ, load_CSV(joinpath(RBInfo.paths.ROM_structures_path, RHSₙi)))
      RBVars.nₛᵘ = size(RBVars.RHSₙ[i])[1]
    else
      @info "Failed to import the block number $i of the reduced affine RHS: must build it"
      push!(operators, "F")
    end

  end

  operators

end

function get_RB_LHS_blocks(RBInfo, RBVars::RBProblem, Param; FEMSpace = nothing)


  if RBInfo.case === 0
    MAₙ_1 = RBVars.Mₙ + RBInfo.δt / 2 * RBVars.Aₙ * Param.μ
    MAₙ_2 = RBVars.Mₙ - RBInfo.δt / 2 * RBVars.Aₙ * Param.μ
  else
    Aₙ_μ = assemble_stiffness(FEMSpace, RBInfo, Param)
    (_, Aₙ_μ_affine) = MDEIM_online(Aₙ_μ, RBVars.Aₙ_affine, RBVars.Aₙ_idx)
    MAₙ = RBVars.Mₙ + 2 / 3 * RBInfo.δt * Aₙ_μ_affine
  end

  Φₜᵘ_1 = RBVars.Φₜᵘ[2:end, :]' * RBVars.Φₜᵘ[1:end - 1, :]

  block1 = zeros(RBVars.nᵘ, RBVars.nᵘ)
  for i_s = 1:RBVars.nₛᵘ
    for i_t = 1:RBVars.nₜᵘ

      i_st = index_mapping(i_s, i_t)

      for j_s = 1:RBVars.nₛᵘ
        for j_t = 1:RBVars.nₜᵘ

          j_st = index_mapping(j_s, j_t)
          RBVars.LHSₙ[1][i_st, j_st] = MAₙ_1[i_s, j_s] * (i_t == j_t)
          + MAₙ_2[i_s, j_s] * Φₜᵘ_1[i_t, j_t]

        end
      end

    end
  end

  push!(RBVars.LHSₙ, block1)
  if RBInfo.save_offline_structures && !RBInfo.problem_nonlinearities["A"]
    save_variable(RBVars.LHSₙ[1], "LHSₙ1", "csv", joinpath(RBInfo.paths.ROM_structures_path, "LHSₙ1"))
  end


end

function get_RB_RHS_blocks(RBInfo, RBVars::RBProblem, Param; FEMSpace = nothing)


  Ffun = assemble_forcing(FEMSpace, Param)
  F = [Ffun(tᵢ) for tᵢ = RBInfo.t₀+RBInfo.δt:RBInfo.δt:RBInfo.T]
  Fₙ = (RBVars.Φₛᵘ)' * F * RBVars.Φₜᵘ
  push!(RBVars.RHSₙ, reshape(Fₙ, :, 1))

  if RBInfo.save_offline_structures && RBInfo.case < 2
    save_variable(RBVars.RHSₙ[1], "RHSₙ1", "csv", joinpath(RBInfo.paths.ROM_structures_path, "RHSₙ1"))
  end

end

function get_RB_system(RBInfo, RBVars::RBProblem, FEMSpace=nothing, Param=nothing)


  @info "Preparing the RB system: fetching online reduced structures"

  operators = check_affine_blocks(RBInfo, RBVars)

  if "A" in operators
    get_RB_LHS_blocks(RBInfo, RBVars, Param; FEMSpace)
  end

  if "F" in operators
    get_RB_RHS_blocks(RBInfo, RBVars, Param; FEMSpace)
  end

end
