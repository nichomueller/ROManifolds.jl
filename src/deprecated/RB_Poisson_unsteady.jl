include("RB_Poisson_steady.jl")

function get_snapshot_matrix(ROM_info, RB_variables::RB_problem)
    #=MODIFY
    =#

    @info "Importing the snapshot matrix, number of snapshots considered: $n_snap"

    var = "uₕ"
    try
        Sᵘ = Matrix(CSV.read(ROM_info.FEM_snap_path * var * ".csv", DataFrame))[:, 1:(ROM_info.nₛ * ROM_info.Nₜ)]
    catch e
        println("Error: $e. Impossible to load the snapshots matrix")
    end

    RB_variables.Sᵘ = Sᵘ
    RB_variables.Nᵤˢ = size(Sᵘ)[1]

    @info "Dimension of snapshot matrix: $(size(Sᵘ)); (Nᵤˢ, Nₜ, nₛ) = ($RB_variables.Nᵤˢ, $ROM_info.Nₜ, $ROM_info.nₛ)"

end

function PODs_time(ROM_info, RB_variables::RB_problem)
    #=MODIFY
    =#

    @info "Performing the temporal POD for field u, using a tolerance of $ROM_info.ϵₜ"

    if ROM_info.time_reduction_technique === "ST-HOSVD"
      Sᵘₜ = zeros(ROM_info.Nₜ, RB_variables.nᵤˢ * ROM_info.nₛ)
      for i in 1:ROM_info.nₛ
        Sᵘₜ[:, RB_variables.nᵤˢ * i:RB_variables.nᵤˢ * (i + 1)] = \
        transpose(Sᵘ[:, ROM_info.Nₜ * i:ROM_info.Nₜ * (i + 1)])
      end
    else
      Sᵘₜ = zeros(ROM_info.Nₜ, RB_variables.Nᵤˢ * ROM_info.nₛ)
      for i in 1:ROM_info.nₛ
          Sᵘₜ[:, RB_variables.Nᵤˢ * i:RB_variables.Nᵤˢ * (i + 1)] = \
          transpose(Sᵘ[:, ROM_info.Nₜ * i:ROM_info.Nₜ * (i + 1)])
      end
    end

    Φₜᵘ = POD(Sᵘ, ROM_info.ϵₜ)
    nₜᵘ = size(Φₜᵘ)[2]

    return (Φₜᵘ, nₜᵘ)

end

function build_reduced_basis(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Building the space-time reduced basis, using tolerances of $(ROM_info.ϵₛ, ROM_info.ϵₜ)"

  RB_building_time = @elapsed begin
    (Φₛᵘ, nₛᵘ) = PODs_space(ROM_info, RB_variables)
    (Φₜᵘ, nₜᵘ) = PODs_time(ROM_info, RB_variables)
  end

  (RB_variables.Φₛᵘ, RB_variables.nₛᵘ) = (Φₛᵘ, nₛᵘ)
  (RB_variables.Φₜᵘ, RB_variables.nₜᵘ) = (Φₜᵘ, nₜᵘ)
  RB_variables.nᵘ = nₛᵘ * nₜᵘ
  RB_variables.offline_time += RB_building_time

  if ROM_info.save_offline_structures
    save_variable(Φₛᵘ, "Φₛᵘ", "csv", joinpath(ROM_info.paths.basis_path, "Φₛᵘ"))
    save_variable(Φₜᵘ, "Φₜᵘ", "csv", joinpath(ROM_info.paths.basis_path, "Φₜᵘ"))
  end

end

function import_reduced_basis(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Importing the reduced basis"

  Φₛᵘ = load_variable("Φₛᵘ", "csv", joinpath(ROM_info.paths.basis_path, "Φₛᵘ"))
  Φₜᵘ = load_variable("Φₜᵘ", "csv", joinpath(ROM_info.paths.basis_path, "Φₜᵘ"))
  RB_variables.Φₛᵘ = Φₛᵘ
  RB_variables.Φₜᵘ = Φₜᵘ
  (RB_variables.Nₛᵘ, RB_variables.nₛᵘ) = size(Φₛᵘ)
  RB_variables.nₜᵘ = size(Φₜᵘ)
  RB_variables.nᵘ = RB_variables.nₛᵘ * RB_variables.nₜᵘ

end

function check_reduced_affine_components(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  operators = []

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Mₙ.csv"))
    @info "Importing reduced mass matrix"
    RB_variables.Mₙ = load_variable("Mₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Mₙ"))
  else
    @info "Failed to import the reduced mass matrix: must build it"
    push!(operators, "M")
  end

  if ROM_info.problem_nonlinearities["A"] === false

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
      @info "Importing reduced affine stiffness matrix"
      RB_variables.Aₙ = load_variable("Aₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ"))
    else
      @info "Failed to import the reduced affine stiffness matrix: must build it"
      push!(operators, "A")
    end

  else

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_affine.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_idx.csv"))
      @info "Importing MDEIM offline structures for the stiffness matrix"
      RB_variables.Aₙ_affine = load_variable("Aₙ_affine", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_affine"))
      RB_variables.Aₙ_idx = load_variable("Aₙ_idx", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_idx"))
    else
      @info "Failed to import MDEIM offline structures for the stiffness matrix: must build them"
      push!(operators, "A")
    end

  end

  if (ROM_info.problem_nonlinearities["f"] === true || ROM_info.problem_nonlinearities["h"] === true)

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
      @info "Importing reduced affine RHS vector"
      RB_variables.Fₙ = load_variable("Fₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ"))
    else
      @info "Failed to import the reduced affine RHS vector: must build it"
      push!(operators, "F")
    end

  else

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_affine.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_idx.csv"))
      @info "Importing DEIM offline structures for the RHS vector"
      RB_variables.Fₙ_affine = load_variable("Fₙ_affine", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_affine"))
      RB_variables.Fₙ_idx = load_variable("Fₙ_idx", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_idx"))
    else
      @info "Failed to import DEIM offline structures for the RHS vector: must build them"
      push!(operators, "F")
    end

  end

  operators

end

function index_mapping(i::T, j::T) where T<:Int64
  #=MODIFY
  =#

  return int(i * nₜᵘ + j)

end

function get_generalized_coordinates(ROM_info, RB_variables::RB_problem, snaps = nothing)
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
        S_ij = RB_variables.Sᵘ[:, i_nₛ * ROM_info.Nₜ:(i_nₛ + 1) * ROM_info.Nₜ]
        û[index_mapping(i_s, i_t), i_nₛ] = Π_ij' * S_ij
      end
    end
  end

  RB_variables.û = û

  if ROM_info.save_offline_structures
    save_variable(û, "û", "csv", joinpath(ROM_info.paths.gen_coords_path, "û"))
  end

end

function initialize_RB_system(RB_variables::RB_problem)
  #=MODIFY
  =#

  RB_variables.LHSₙ[1] = zeros(RB_variables.nᵘ, RB_variables.nᵘ)
  RB_variables.RHSₙ[1] = zeros(RB_variables.nᵘ)

end

function check_affine_blocks(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  if ROM_info.import_offline_structures === false
    return ["A", "F"]
  end

  operators = []

  for i = 1:1

    LHSₙi = "LHSₙ" * string(i)
    RHSₙi = "RHSₙ" * string(i)

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, LHSₙi * ".csv"))
      @info "Importing block number $i of the reduced affine LHS"
      RB_variables.LHSₙ[i] = load_variable(LHSₙi, "csv", joinpath(ROM_info.paths.ROM_structures_path, LHSₙi))
      RB_variables.nₛᵘ = size(RB_variables.LHSₙ[i])[1]
    else
      @info "Failed to import the block number $i of the reduced affine LHS: must build it"
      push!(operators, "A")
    end

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, RHSₙi * ".csv"))
      @info "Importing block number $i of the reduced affine RHS"
      RB_variables.RHSₙ[i] = load_variable(RHSₙi, "csv", joinpath(ROM_info.paths.ROM_structures_path, RHSₙi))
      RB_variables.nₛᵘ = size(RB_variables.RHSₙ[i])[1]
    else
      @info "Failed to import the block number $i of the reduced affine RHS: must build it"
      push!(operators, "F")
    end

  end

  operators

end

function get_RB_system(ROM_info, RB_variables::RB_problem, FE_space = nothing, param = nothing)
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
