include("../utils/general.jl")
include("RB_utils.jl")


function get_snapshot_matrix(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Importing the snapshot matrix, number of snapshots considered: $(ROM_info.nₛ)"

  var = "uₕ"
  Sᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, var * ".csv"), DataFrame))[:, 1:ROM_info.nₛ]
  #= try
    Sᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, var * ".csv"), DataFrame))[:, 1:ROM_info.nₛ]
  catch e
    println("Error: $e. Impossible to load the snapshots matrix")
  end =#

  @info "Dimension of snapshot matrix: $(size(Sᵘ))"

  RB_variables.Sᵘ = Sᵘ
  RB_variables.Nᵤˢ = size(Sᵘ)[1]

end


function get_norm_matrix(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Importing the norm matrix"

  if check_norm_matrix(RB_variables)

    Xᵘ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Xᵘ.csv"); convert_to_sparse = true)
    #= try
      Xᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_structures_path, "Xᵘ.csv"), DataFrame))
      RB_variables.Xᵘ = Xᵘ
    catch e
      println("Error: $e. Impossible to load the H1 norm matrix")
    end =#

  end

end


function check_norm_matrix(RB_variables::RB_problem)
  #=MODIFY
  =#

  isempty(RB_variables.Xᵘ) || maximum(abs.(RB_variables.Xᵘ)) === 0

end


#= function preprocess()
  #=MODIFY
  =#

end =#


function set_to_zero_RB_times(RB_variables::RB_problem)
  #=MODIFY
  =#

  RB_variables.offline_time = 0.0
  RB_variables.online_time = 0.0

end


function PODs_space(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Performing the spatial POD for field u, using a tolerance of $ROM_info.ϵˢ"

  get_norm_matrix(ROM_info, RB_variables)
  Φₛᵘ = POD(RB_variables.Sᵘ, ROM_info.ϵˢ, RB_variables.Xᵘ)
  nₛᵘ = size(Φₛᵘ)[2]

  return (Φₛᵘ, nₛᵘ)

end


function build_reduced_basis(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Building the reduced basis, using a tolerance of $(ROM_info.ϵˢ)"

  RB_building_time = @elapsed begin
    (Φₛᵘ, nₛᵘ) = PODs_space(ROM_info, RB_variables)
  end

  (RB_variables.Φₛᵘ, RB_variables.nₛᵘ) = (Φₛᵘ, nₛᵘ)
  RB_variables.offline_time += RB_building_time

  if ROM_info.save_offline_structures
    save_variable(Φₛᵘ, "Φₛᵘ", "csv", joinpath(ROM_info.paths.basis_path, "Φₛᵘ"))
  end

end


function import_reduced_basis(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Importing the reduced basis"

  Φₛᵘ = load_CSV(joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
  RB_variables.Φₛᵘ = Φₛᵘ
  (RB_variables.Nₛᵘ, RB_variables.nₛᵘ) = size(Φₛᵘ)

end


function check_reduced_affine_components(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  operators = []

  if ROM_info.problem_nonlinearities["A"] === false

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
      @info "Importing reduced affine stiffness matrix"
      RB_variables.Aₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
      RB_variables.nₛᵘ = size(RB_variables.Aₙ)[1]
    else
      @info "Failed to import the reduced affine stiffness matrix: must build it"
      push!(operators, "A")
    end

  else

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_affine.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_idx.csv"))
      @info "Importing MDEIM offline structures for the stiffness matrix"
      RB_variables.Aₙ_affine = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_affine.csv"))
      RB_variables.Aₙ_idx = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_idx.csv"))
      RB_variables.nₛᵘ = size(RB_variables.Aₙ_affine)[1]
    else
      @info "Failed to import MDEIM offline structures for the stiffness matrix: must build them"
      push!(operators, "A")
    end

  end

  if (ROM_info.problem_nonlinearities["f"] === true || ROM_info.problem_nonlinearities["h"] === true)

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
      @info "Importing reduced affine RHS vector"
      RB_variables.Fₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
      RB_variables.nₛᵘ = size(RB_variables.Fₙ)[1]
    else
      @info "Failed to import the reduced affine RHS vector: must build it"
      push!(operators, "F")
    end

  else

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_affine.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_idx.csv"))
      @info "Importing DEIM offline structures for the RHS vector"
      RB_variables.Fₙ_affine = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_affine.csv"))
      RB_variables.Fₙ_idx = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_idx.csv"))
      RB_variables.nₛᵘ = size(RB_variables.Fₙ_affine)[1]
    else
      @info "Failed to import DEIM offline structures for the RHS vector: must build them"
      push!(operators, "F")
    end

  end

  operators

end


#= function get_reduced_affine_components(ROM_info, RB_variables)
    #=MODIFY
    =#

    @info "Importing affine stiffness matrix"

    if ROM_info.problem_nonlinearities["A"] === false
        RB_variables.Aₙ = load_variable("Aₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ"))
    else
        RB_variables.Aₙ_affine = load_variable("Aₙ_mdeim", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_mdeim"))
        RB_variables.Aₙ_idx = load_variable("Aₙ_mdeim_idx", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_mdeim_idx"))
    end

    @info "Importing affine forcing term"

    if ROM_info.problem_nonlinearities["F"] === false
        RB_variables.Fₙ = load_variable("Fₙ", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ"))
    else
        RB_variables.Fₙ_affine = load_variable("Fₙ_deim", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_deim"))
        RB_variables.Fₙ_idx = load_variable("Fₙ_deim_idx", "csv", joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_deim_idx"))
    end


end =#


#= function check_and_return_DEIM_MDEIM(ROM_info, RB_variables)
    #=MODIFY
    =#

    if ROM_info.problem_nonlinearities["A"] === true && (isempty(RB_variables.Aₙ_affine) || maximum(abs.(RB_variables.Aₙ_affine)) === 0)
        if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_mdeim")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_idx"))
            get_reduced_affine_components(ROM_info, RB_variables)
        else
            assemble_reduced_affine_components(ROM_info, RB_variables)
        end

        if (ROM_info.problem_nonlinearities["f"] === true || ROM_info.problem_nonlinearities["h"] === true) && (isempty(RB_variables.Fₙ_affine) || maximum(abs.(RB_variables.Fₙ_affine)) === 0)
            get_reduced_affine_components(ROM_info, RB_variables)
        else
            assemble_reduced_affine_components(ROM_info, RB_variables)
        end

    end

end =#


function get_generalized_coordinates(ROM_info, RB_variables::RB_problem, snaps=nothing)
  #=MODIFY
  =#

  get_norm_matrix(ROM_info, RB_variables)

  if snaps === nothing || maximum(snaps) > ROM_info.nₛ
    snaps = 1:ROM_info.nₛ
  end

  û = zeros(RB_variables.nₛᵘ, length(snaps))
  Φₛᵘ_normed = RB_variables.Xᵘ * RB_variables.Φₛᵘ
  RB_variables.û = RB_variables.Sᵘ * Φₛᵘ_normed

  if ROM_info.save_offline_structures
    save_variable(RB_variables.û, "û", "csv", joinpath(ROM_info.paths.gen_coords_path, "û"))
  end

end


function initialize_RB_system(RB_variables::RB_problem)
  #=MODIFY
  =#

  RB_variables.LHSₙ[1] = zeros(RB_variables.nₛᵘ, RB_variables.nₛᵘ)
  RB_variables.RHSₙ[1] = zeros(RB_variables.nₛᵘ)

end


function get_RB_system(ROM_info, RB_variables::RB_problem, param; FE_space=nothing)
  #=MODIFY
  =#

  @info "Preparing the RB system: fetching online reduced structures"

  if ROM_info.problem_nonlinearities["Aₙ"] === false
    RB_variables.LHSₙ[1] = param.α(0) * RB_variables.Aₙ
  else
    A_μ = assemble_stiffness(FE_space, ROM_info, param)
    (_, A_μ_affine) = MDEIM_online(A_μ, RB_variables.Aₙ_affine, RB_variables.Aₙ_idx)
    RB_variables.LHSₙ[1] = A_μ_affine
  end

  if ROM_info.problem_nonlinearities["f"] === false && ROM_info.problem_nonlinearities["h"] === false
    RB_variables.RHSₙ[1] = RB_variables.Fₙ
  else
    F_μ = assemble_forcing(FE_space, param)
    (_, F_μ_affine) = DEIM_online(F_μ, RB_variables.Fₙ_affine, RB_variables.Fₙ_idx)
    RB_variables.RHSₙ[1] = F_μ_affine
  end

end


function solve_RB_system(ROM_info, RB_variables::RB_problem, param; FE_space=nothing)
  #=MODIFY
  =#

  if ROM_info.case > 0 && FE_space === nothing

    @error "Provide a valid FE_space struct when A, F are parameter-dependent"

  end

  get_RB_system(ROM_info, RB_variables, param; FE_space)

  @info "Solving RB problem via backslash"
  @info "Condition number of the system's matrix: $(cond(RB_variables.LHSₙ[1]))"
  RB_variables.uₙ = RB_variables.LHSₙ[1] \ RB_variables.RHSₙ[1]

end


function reconstruct_FEM_soluiton(RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Reconstructing FEM solution from the newly computed RB one"

  RB_variables.ũ = RB_variables.Φₛᵘ * RB_variables.uₙ

end

function build_RB_approximation(ROM_info, RB_variables::RB_problem; μ=nothing)
  #=MODIFY
  =#

  @info "Building $(ROM_info.RB_method) approximation with $(ROM_info.nₛ) snapshots and tolerances of $(ROM_info.ϵˢ) in space"

  if ROM_info.import_snapshots
    get_snapshot_matrix(ROM_info, RB_variables)
    import_snapshots_success = true
  else
    import_snapshots_success = false
  end

  if ROM_info.import_offline_structures
    import_reduced_basis(ROM_info, RB_variables)
    import_basis_success = true
  else
    import_basis_success = false
  end

  if !import_snapshots_success && !import_basis_success
    @error "Impossible to assemble the reduced problem if neither the snapshots nor the bases can be loaded"
  end

  if import_snapshots_success && !import_basis_success
    @info "Failed to import the reduced basis, building it via POD"
    build_reduced_basis(ROM_info, RB_variables)
  end

  if ROM_info.import_offline_structures
    operators = check_reduced_affine_components(ROM_info, RB_variables)
    if !isempty(operators)
      assemble_reduced_affine_components(ROM_info, RB_variables, operators; μ=μ)
    end
  else
    assemble_reduced_affine_components(ROM_info, RB_variables; μ=μ)
  end

end


function testing_phase(ROM_info, RB_variables::RB_problem, μ, param_nbs; FE_space=nothing)
  #=MODIFY
  =#

  mean_H1_err = 0.0
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0
  set_to_zero_RB_times(RB_variables)
  get_norm_matrix(ROM_info, RB_variables)

  ũ_μ = zeros(RB_variables.Nᵤˢ)
  uₙ_μ = zeros(RB_variables.nᵤˢ)

  for μ_nb in μ[param_nbs]
    @info "Considering parameter number: $μ_nbs"

    parametric_info = get_parametric_specifics(ROM_info, μ_nb)
    if ROM_info.case === 3
      FE_space = FE_space(problem_info, parametric_info)
    end

    uₕ_test = Matrix(CSV.read(joinpath(ROM_info.path.FEM_snap_path, "uₕ.csv"), DataFrame))[:, μ_nb]

    online_time = @elapsed begin
      solve_RB_system(ROM_info, RB_variables, parametric_info, FE_space=FE_space)
    end
    reconstruction_time = @elapsed begin
      reconstruct_FEM_soluiton(RB_variables)
    end
    mean_online_time = online_time / length(μ_nbs)
    mean_reconstruction_time = reconstruction_time / length(μ_nbs)

    H1_err_nb = compute_errors(RB_variables, uₕ_test)
    mean_H1_err += H1_err_nb / length(μ_nbs)

    ũ_μ = hcat(ũ_μ, RB_variables.ũ)
    uₙ_μ = hcat(uₙ_μ, RB_variables.uₙ)

    @info "Online wall time: $online_time s (snapshot number $μ_nb)"
    @info "Relative reconstruction H1-error: $H1_err_nb (snapshot number $μ_nb)"

  end

  string_μ_nbs = "params"
  for μ_nb in μ_nbs
    string_μ_nbs *= "_" * string(μ_nb)
  end

  if ROM_info.save_results

    if !ROM_info.import_offline_structures
      save_variable(RB_variables.offline_time, "offline_time", "csv", ROM_info.paths.results_path)
    end

    path = joinpath(ROM_info.paths.results_path, string_μ_nbs)
    save_variable(ũ_μ, "ũ", "csv", path)
    save_variable(uₙ_μ, "uₙ", "csv", path)
    save_variable(mean_H1_err, "mean_H1_err", "csv", path)
    save_variable(mean_online_time, "mean_online_time", "csv", path)
    save_variable(mean_reconstruction_time, "mean_reconstruction_time", "csv", path)

  end

end


function compute_errors(RB_variables::RB_problem, uₕ_test)
  #=MODIFY
  =#

  mynorm(uₕ_test - RB_variables.ũ, RB_variables.Xᵘ) / mynorm(uₕ_test, RB_variables.Xᵘ)

end
