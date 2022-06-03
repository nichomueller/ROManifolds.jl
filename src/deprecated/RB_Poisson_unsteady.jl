include("RB_Poisson_steady.jl")

function get_snapshot_matrix(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  @info "Importing the snapshot matrix, number of snapshots considered: $n_snap"

  name = "uₕ"
  Sᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, name * ".csv"), DataFrame))[:, 1:(ROM_info.nₛ*ROM_info.Nₜ)]
  #= try
      Sᵘ = Matrix(CSV.read(ROM_info.FEM_snap_path * var * ".csv", DataFrame))[:, 1:(ROM_info.nₛ * ROM_info.Nₜ)]
  catch e
      println("Error: $e. Impossible to load the snapshots matrix")
  end =#

  RB_variables.Sᵘ = Sᵘ
  RB_variables.Nᵤˢ = size(Sᵘ)[1]

  @info "Dimension of snapshot matrix: $(size(Sᵘ)); (Nᵤˢ, Nₜ, nₛ) = ($RB_variables.Nᵤˢ, $ROM_info.Nₜ, $ROM_info.nₛ)"

end

function PODs_time(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  @info "Performing the temporal POD for field u, using a tolerance of $ROM_info.ϵₜ"

  if ROM_info.time_reduction_technique === "ST-HOSVD"
    Sᵘₜ = zeros(ROM_info.Nₜ, RB_variables.nᵤˢ * ROM_info.nₛ)
    for i in 1:ROM_info.nₛ
      Sᵘₜ[:, RB_variables.nᵤˢ*i:RB_variables.nᵤˢ*(i+1)] = \
      transpose(Sᵘ[:, ROM_info.Nₜ*i:ROM_info.Nₜ*(i+1)])
    end
  else
    Sᵘₜ = zeros(ROM_info.Nₜ, RB_variables.Nᵤˢ * ROM_info.nₛ)
    for i in 1:ROM_info.nₛ
      Sᵘₜ[:, RB_variables.Nᵤˢ*i:RB_variables.Nᵤˢ*(i+1)] = \
      transpose(Sᵘ[:, ROM_info.Nₜ*i:ROM_info.Nₜ*(i+1)])
    end
  end

  Φₜᵘ, _ = POD(Sᵘ, ROM_info.ϵₜ)
  RB_variables.Φₜᵘ = Φₜᵘ
  RB_variables.nₜᵘ = size(Φₛᵘ)[2]

end

function build_reduced_basis(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  @info "Building the space-time reduced basis, using tolerances of $(ROM_info.ϵₛ, ROM_info.ϵₜ)"

  RB_building_time = @elapsed begin
    PODs_space(ROM_info, RB_variables)
    PODs_time(ROM_info, RB_variables)
  end

  RB_variables.nᵘ = nₛᵘ * nₜᵘ
  RB_variables.offline_time += RB_building_time

  if ROM_info.save_offline_structures
    save_CSV(Φₛᵘ, joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
    save_CSV(Φₜᵘ, joinpath(ROM_info.paths.basis_path, "Φₜᵘ.csv"))
  end

end

function import_reduced_basis(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  @info "Importing the reduced basis"

  RB_variables.Φₛᵘ = load_CSV(joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
  RB_variables.Φₜᵘ = load_CSV(joinpath(ROM_info.paths.basis_path, "Φₜᵘ.csv"))
  (RB_variables.Nₛᵘ, RB_variables.nₛᵘ) = size(RB_variables.Φₛᵘ)
  RB_variables.nₜᵘ = size(RB_variables.Φₜᵘ)[2]
  RB_variables.nᵘ = RB_variables.nₛᵘ * RB_variables.nₜᵘ

end

function check_reduced_affine_components(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  operators = []

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    @info "Importing reduced mass matrix"
    RB_variables.Mₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
  else
    @info "Failed to import the reduced mass matrix: must build it"
    push!(operators, "M")
  end

  push!(operators, check_reduced_affine_components(ROM_info, RB_variables.S))

  operators

end

function index_mapping(i::T, j::T) where {T<:Int64}
  #=MODIFY
  =#

  return int(i * nₜᵘ + j)

end

function get_generalized_coordinates(ROM_info, RB_variables::RBProblem, snaps=nothing)
  #=MODIFY
  =#

  if !check_norm_matrix(RB_variables)
    get_norm_matrix(ROM_info, RB_variables)
  end

  if snaps === nothing || maximum(snaps) > ROM_info.nₛ
    snaps = 1:ROM_info.nₛ
  end

  û = zeros(RB_variables.nᵘ, length(snaps))
  Φₛᵘ_normed = RB_variables.Xᵘ * RB_variables.Φₛᵘ

  for i_nₛ = snaps
    @info "Assembling generalized coordinate relative to snapshot $(i_nₛ)"
    for i_s = 1:RB_variables.nₛᵘ
      for i_t = 1:RB_variables.nₛᵗ
        Π_ij = kron(Φₛᵘ_normed[:, i_s], RB_variables.Φₜᵘ[:, i_t])
        S_ij = RB_variables.Sᵘ[:, (i_nₛ-1)*ROM_info.Nₜ+1:i_nₛ*ROM_info.Nₜ]
        û[index_mapping(i_s, i_t), i_nₛ] = Π_ij' * S_ij
      end
    end
  end

  RB_variables.û = û

  if ROM_info.save_offline_structures
    save_CSV(û, joinpath(ROM_info.paths.gen_coords_path, "û.csv"))
  end

end

function check_affine_blocks(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  if ROM_info.import_offline_structures === false
    return ["A", "F"]
  end

  operators = []
  initialize_RB_system(RB_variables)

  for i = 1:1

    LHSₙi = "LHSₙ" * string(i) * ".csv"
    RHSₙi = "RHSₙ" * string(i) * ".csv"

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, LHSₙi * ".csv"))
      @info "Importing block number $i of the reduced affine LHS"
      push!(RB_variables.LHSₙ, load_CSV(joinpath(ROM_info.paths.ROM_structures_path, LHSₙi)))
      RB_variables.nₛᵘ = size(RB_variables.LHSₙ[i])[1]
    else
      @info "Failed to import the block number $i of the reduced affine LHS: must build it"
      push!(operators, "A")
    end

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, RHSₙi * ".csv"))
      @info "Importing block number $i of the reduced affine RHS"
      push!(RB_variables.RHSₙ, load_CSV(joinpath(ROM_info.paths.ROM_structures_path, RHSₙi)))
      RB_variables.nₛᵘ = size(RB_variables.RHSₙ[i])[1]
    else
      @info "Failed to import the block number $i of the reduced affine RHS: must build it"
      push!(operators, "F")
    end

  end

  operators

end

function get_RB_LHS_blocks(ROM_info, RB_variables::RBProblem, param; FE_space = nothing)
  #=MODIFY
  =#

  if ROM_info.case === 0
    MAₙ_1 = RB_variables.Mₙ + ROM_info.δt / 2 * RB_variables.Aₙ * param.μ
    MAₙ_2 = RB_variables.Mₙ - ROM_info.δt / 2 * RB_variables.Aₙ * param.μ
  else
    Aₙ_μ = assemble_stiffness(FE_space, ROM_info, param)
    (_, Aₙ_μ_affine) = MDEIM_online(Aₙ_μ, RB_variables.Aₙ_affine, RB_variables.Aₙ_idx)
    MAₙ = RB_variables.Mₙ + 2 / 3 * ROM_info.δt * Aₙ_μ_affine
  end

  Φₜᵘ_1 = RB_variables.Φₜᵘ[2:end, :]' * RB_variables.Φₜᵘ[1:end - 1, :]

  block1 = zeros(RB_variables.nᵘ, RB_variables.nᵘ)
  for i_s = 1:RB_variables.nₛᵘ
    for i_t = 1:RB_variables.nₜᵘ

      i_st = index_mapping(i_s, i_t)

      for j_s = 1:RB_variables.nₛᵘ
        for j_t = 1:RB_variables.nₜᵘ

          j_st = index_mapping(j_s, j_t)
          RB_variables.LHSₙ[1][i_st, j_st] = MAₙ_1[i_s, j_s] * (i_t == j_t)
          + MAₙ_2[i_s, j_s] * Φₜᵘ_1[i_t, j_t]

        end
      end

    end
  end

  push!(RB_variables.LHSₙ, block1)
  if ROM_info.save_offline_structures && !ROM_info.problem_nonlinearities["A"]
    save_variable(RB_variables.LHSₙ[1], "LHSₙ1", "csv", joinpath(ROM_info.paths.ROM_structures_path, "LHSₙ1"))
  end


end

function get_RB_RHS_blocks(ROM_info, RB_variables::RBProblem, param; FE_space = nothing)
  #=MODIFY
  =#

  Ffun = assemble_forcing(FE_space, param)
  F = [Ffun(tᵢ) for tᵢ = ROM_info.t₀+ROM_info.δt:ROM_info.δt:ROM_info.T]
  Fₙ = (RB_variables.Φₛᵘ)' * F * RB_variables.Φₜᵘ
  push!(RB_variables.RHSₙ, reshape(Fₙ, :, 1))

  if ROM_info.save_offline_structures && ROM_info.case < 2
    save_variable(RB_variables.RHSₙ[1], "RHSₙ1", "csv", joinpath(ROM_info.paths.ROM_structures_path, "RHSₙ1"))
  end

end

function get_RB_system(ROM_info, RB_variables::RBProblem, FE_space=nothing, param=nothing)
  #=MODIFY
  =#

  @info "Preparing the RB system: fetching online reduced structures"

  operators = check_affine_blocks(ROM_info, RB_variables)

  if "A" in operators
    get_RB_LHS_blocks(ROM_info, RB_variables, param; FE_space)
  end

  if "F" in operators
    get_RB_RHS_blocks(ROM_info, RB_variables, param; FE_space)
  end

end
