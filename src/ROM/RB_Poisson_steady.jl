include("../utils/general.jl")
include("RB_utils.jl")
include("../FEM/FEM.jl")

function get_snapshot_matrix(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Importing the snapshot matrix, number of snapshots considered: $(ROM_info.nₛ)"

  name = "uₕ"
  Sᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, name * ".csv"), DataFrame))[:, 1:ROM_info.nₛ]
  #= try
    Sᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, var * ".csv"), DataFrame))[:, 1:ROM_info.nₛ]
  catch e
    println("Error: $e. Impossible to load the snapshots matrix")
  end =#

  @info "Dimension of snapshot matrix: $(size(Sᵘ))"

  RB_variables.Sᵘ = Sᵘ
  RB_variables.Nₛᵘ = size(Sᵘ)[1]

end


function get_norm_matrix(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  if check_norm_matrix(RB_variables)

    @info "Importing the norm matrix"

    Xᵘ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Xᵘ.csv"); convert_to_sparse = true)
    RB_variables.Xᵘ = Xᵘ
    #= try
      Xᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_structures_path, "Xᵘ.csv"), DataFrame))
      RB_variables.Xᵘ = Xᵘ
    catch e
      println("Error: $e. Impossible to load the H1 norm matrix")
    end =#

    @info "Dimension of norm matrix: $(size(RB_variables.Xᵘ))"

  end

end


function check_norm_matrix(RB_variables::RB_problem)
  #=MODIFY
  =#

  isempty(RB_variables.Xᵘ) || maximum(abs.(RB_variables.Xᵘ)) === 0

end

function get_inverse_P_matrix(ROM_info, RB_variables::PoissonSTPGRB)
  #=MODIFY
  =#

  if isempty(RB_variables.Pᵘ_inv) || maximum(abs.(RB_variables.Pᵘ_inv)) === 0
    @info "Building the inverse of the diag preconditioner of the H1 norm matrix, ST-PGRB method"

    if isfile(joinpath(ROM_info.paths.FEM_structures_path, "Pᵘ_inv.csv"))
      RB_variables.Pᵘ_inv = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_structures_path, "Pᵘ_inv.csv"), DataFrame))
    else
      get_norm_matrix(ROM_info, RB_variables)
      Pᵘ = diag(RB_variables.Xᵘ)
      RB_variables.Pᵘ_inv = I(size(RB_variables.Xᵘ)[1]) \ Pᵘ
      save_CSV(RB_variables.Pᵘ_inv, joinpath(ROM_info.paths.FEM_structures_path, "Pᵘ_inv.csv"))
    end

  end

end

function PODs_space(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Performing the spatial POD for field u, using a tolerance of $(ROM_info.ϵₛ)"

  get_norm_matrix(ROM_info, RB_variables)
  Φₛᵘ = POD(RB_variables.Sᵘ, ROM_info.ϵₛ)
  #Φₛᵘ = POD(RB_variables.Sᵘ, ROM_info.ϵₛ, RB_variables.Xᵘ) SOMETHING WRONG HERE...FIX NORM MATRIX!

  RB_variables.Φₛᵘ = Φₛᵘ
  RB_variables.nₛᵘ = size(Φₛᵘ)[2]

end


function build_reduced_basis(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Building the reduced basis, using a tolerance of $(ROM_info.ϵₛ)"

  RB_building_time = @elapsed begin
    PODs_space(ROM_info, RB_variables)
  end

  RB_variables.offline_time += RB_building_time

  if ROM_info.save_offline_structures
    save_CSV(RB_variables.Φₛᵘ, joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
  end

end


function import_reduced_basis(ROM_info, RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Importing the reduced basis"

  RB_variables.Φₛᵘ = load_CSV(joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
  (RB_variables.Nₛᵘ, RB_variables.nₛᵘ) = size(RB_variables.Φₛᵘ)

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

  Φₛᵘ_normed = RB_variables.Xᵘ * RB_variables.Φₛᵘ
  RB_variables.û = RB_variables.Sᵘ * Φₛᵘ_normed

  if ROM_info.save_offline_structures
    save_CSV(RB_variables.û, joinpath(ROM_info.paths.gen_coords_path, "û.csv"))
  end

end

function assemble_reduced_affine_components(ROM_info, RB_variables::PoissonSTGRB, operators=nothing; μ=nothing)
  #=MODIFY
  =#

  if operators === nothing
    operators = ["A", "F"]
  end

  if "A" in operators

    if ROM_info.problem_nonlinearities["A"] === false

      @info "Assembling affine reduced stiffness"
      projection_time = @elapsed begin
        A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
        RB_variables.Aₙ = (RB_variables.Φₛᵘ)' * A * RB_variables.Φₛᵘ
        if ROM_info.save_offline_structures
          save_CSV(RB_variables.Aₙ, joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
        end
      end

    else

      @info "The stiffness is non-affine: running the MDEIM offline phase"
      projection_time = @elapsed begin
        Aₙ_i = sparse([], [], [])
        for i_nₛ = 1:maximum(10, ROM_info.nₛ)
          parametric_info = get_parametric_specifics(ROM_info, μ[i_nₛ])
          A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
          Aₙ_i = hcat(Aₙ_i, (RB_variables.Φₛᵘ)' * A_i * RB_variables.Φₛᵘ)
        end
        Aₙ_i = reshape(Aₙ_i, :, 1)
        if ROM_info.save_offline_structures
          (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵₛ, joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_mdeim"))
        else
          (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵₛ)
        end
      end

    end

  end

  if "F" in operators

    if ROM_info.problem_nonlinearities["f"] === false || ROM_info.problem_nonlinearities["h"] === false

      @info "Assembling affine reduced forcing term"
      projection_time += @elapsed begin
        F = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "F.csv"))
        RB_variables.Fₙ = (RB_variables.Φₛᵘ)' * F
        if ROM_info.save_offline_structures
          save_CSV(RB_variables.Fₙ, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
        end
      end

    else

      @info "The forcing term is non-affine: running the DEIM offline phase"
      projection_time += @elapsed begin
        Fₙ_i = Float64[]
        for i_nₛ = 1:maximum(10, ROM_info.nₛ)
          parametric_info = get_parametric_specifics(ROM_info, μ[i_nₛ])
          F_i = assemble_forcing(FE_space, parametric_info)
          Fₙ_i = hcat(Fₙ_i, (RB_variables.Φₛᵘ)' * F_i)
        end
        if ROM_info.save_offline_structures
          (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵₛ, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_deim"))
        else
          (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵₛ)
        end
      end
    end

  end

  RB_variables.offline_time += projection_time

end

function assemble_reduced_affine_components(ROM_info, RB_variables::PoissonSTPGRB, operators=nothing; μ=nothing)
  #=MODIFY
  =#

  if isempty(RB_variables.Φₛᵘ) || maximum(abs.(RB_variables.Φₛᵘ)) === 0
    @error "Error: must generate or import spatial RBs before computing the reduced affine components"
  end

  if isnothing(operators)
    operators = ["A", "F"]
  end

  if "A" in operators

    if !ROM_info.problem_nonlinearities["A"]

      @info "Assembling affine reduced stiffness"
      projection_time = @elapsed begin
        A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"))
        AΦₛᵘ = A * RB_variables.Φₛᵘ
        RB_variables.Aₙ = (AΦₛᵘ)' * RB_variables.Pᵘ_inv * AΦₛᵘ
        if ROM_info.save_offline_structures
          save_CSV(RB_variables.Aₙ, joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
        end
      end

    else

      @info "The stiffness is non-affine: running the MDEIM offline phase"
      projection_time = @elapsed begin
        Aₙ_i = sparse([], [], [])
        for i_nₛ = 1:maximum(10, ROM_info.nₛ)
          parametric_info = get_parametric_specifics(ROM_info, μ[i_nₛ])
          A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
          A_iΦₛᵘ = A_i * RB_variables.Φₛᵘ
          Aₙ_i = hcat(Aₙ_i, (A_iΦₛᵘ)' * RB_variables.Pᵘ_inv * A_iΦₛᵘ)
        end
        Aₙ_i = reshape(Aₙ_i, :, 1)
        if ROM_info.save_offline_structures
          (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵₛ, joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_mdeim"))
        else
          (RB_variables.Aₙ_affine, RB_variables.Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵₛ)
        end
      end

    end

  end

  if "F" in operators

    if !ROM_info.problem_nonlinearities["f"] && !ROM_info.problem_nonlinearities["h"]

      @info "Assembling affine reduced forcing term"
      projection_time += @elapsed begin
        F = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "F.csv"))

        if !ROM_info.problem_nonlinearities["A"]

          A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
          AΦₛᵘ = A * RB_variables.Φₛᵘ
          RB_variables.Fₙ = (AΦₛᵘ)' * RB_variables.Pᵘ_inv * F
          if ROM_info.save_offline_structures
            save_CSV(RB_variables.Fₙ, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
          end

        else

          Fₙ_i = Float64[]
          for i_nₛ = 1:maximum(10, ROM_info.nₛ)
            parametric_info = get_parametric_specifics(ROM_info, μ[i_nₛ])
            A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
            A_iΦₛᵘ = A_i * RB_variables.Φₛᵘ
            Fₙ_i = hcat(Fₙ_i, (A_iΦₛᵘ)' * RB_variables.Pᵘ_inv * F)
          end
          if ROM_info.save_offline_structures
            (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵₛ, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_deim"))
          else
            (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵₛ)
          end

        end

      end

    else

      @info "The forcing term is non-affine: running the DEIM offline phase"
      projection_time += @elapsed begin
        Fₙ_i = Float64[]

        if !ROM_info.problem_nonlinearities["A"]

          A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
          AΦₛᵘPᵘ_inv = (A * Φₛᵘ)' * RB_variables.Pᵘ_inv
          for i_nₛ = 1:maximum(10, ROM_info.nₛ)
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
            F_i = assemble_forcing(FE_space, parametric_info)
            Fₙ_i = hcat(Fₙ_i, AΦₛᵘPᵘ_inv * F_i)
          end

        else

          for i_nₛ = 1:maximum(10, ROM_info.nₛ)
            parametric_info = compute_parametric_info(problem_nonlinearities, params, i_nₛ)
            A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
            F_i = assemble_forcing(FE_space, parametric_info)
            A_iΦₛᵘ = A_i * RB_variables.Φₛᵘ
            Fₙ_i = hcat(Fₙ_i, (A_iΦₛᵘ)' * RB_variables.Pᵘ_inv * F_i)
          end

        end

        if ROM_info.save_offline_structures
          (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵₛ, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_deim"))
        else
          (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵₛ)
        end

      end

    end

  end

  RB_variables.offline_time += projection_time

end

#= function initialize_RB_system(RB_variables::RB_problem)
  #=MODIFY
  =#

  RB_variables.LHSₙ[1] = zeros(RB_variables.nₛᵘ, RB_variables.nₛᵘ)
  RB_variables.RHSₙ[1] = zeros(RB_variables.nₛᵘ)

end =#

function get_RB_system(ROM_info, RB_variables::RB_problem, param; FE_space=nothing)
  #=MODIFY
  =#

  @info "Preparing the RB system: fetching online reduced structures"

  if ROM_info.problem_nonlinearities["A"] === false
    push!(RB_variables.LHSₙ, param.α(Point(0.,0.)) * RB_variables.Aₙ)
  else
    A_μ = assemble_stiffness(FE_space, ROM_info, param)
    (_, A_μ_affine) = MDEIM_online(A_μ, RB_variables.Aₙ_affine, RB_variables.Aₙ_idx)
    push!(RB_variables.LHSₙ, A_μ_affine)
  end

  if ROM_info.problem_nonlinearities["f"] === false && ROM_info.problem_nonlinearities["h"] === false
    push!(RB_variables.RHSₙ, RB_variables.Fₙ)
  else
    F_μ = assemble_forcing(FE_space, param)
    (_, F_μ_affine) = DEIM_online(F_μ, RB_variables.Fₙ_affine, RB_variables.Fₙ_idx)
    push!(RB_variables.RHSₙ, F_μ_affine)
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
  RB_variables.uₙ = zeros(RB_variables.nₛᵘ)
  RB_variables.uₙ = RB_variables.LHSₙ[1] \ RB_variables.RHSₙ[1]

end

function reconstruct_FEM_solution(RB_variables::RB_problem)
  #=MODIFY
  =#

  @info "Reconstructing FEM solution from the newly computed RB one"

  RB_variables.ũ = zeros(RB_variables.Nₛᵘ)
  RB_variables.ũ = RB_variables.Φₛᵘ * RB_variables.uₙ

end

function build_RB_approximation(ROM_info, RB_variables::RB_problem; μ=nothing)
  #=MODIFY
  =#

  @info "Building $(ROM_info.RB_method) approximation with $(ROM_info.nₛ) snapshots and tolerances of $(ROM_info.ϵₛ) in space"

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

  get_norm_matrix(ROM_info, RB_variables)

  ũ_μ = zeros(RB_variables.Nₛᵘ, length(param_nbs))
  uₙ_μ = zeros(RB_variables.nₛᵘ, length(param_nbs))

  for nb in param_nbs
    @info "Considering parameter number: $nb"

    #= try
      μ_nb = μ[:, nb]
    catch
      μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
    end =#
    μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))

    parametric_info = get_parametric_specifics(ROM_info, μ_nb)
    if ROM_info.case > 0
      FE_space = FE_space(problem_info, parametric_info)
    end

    uₕ_test = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, nb]

    online_time = @elapsed begin
      solve_RB_system(ROM_info, RB_variables, parametric_info; FE_space = FE_space)
    end
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RB_variables)
    end
    mean_online_time = online_time / length(param_nbs)
    mean_reconstruction_time = reconstruction_time / length(param_nbs)

    H1_err_nb = compute_errors(RB_variables, uₕ_test)
    mean_H1_err += H1_err_nb / length(param_nbs)

    ũ_μ[:, nb - param_nbs[1] + 1] = RB_variables.ũ
    uₙ_μ[:, nb - param_nbs[1] + 1] = RB_variables.uₙ

    @info "Online wall time: $online_time s (snapshot number $μ_nb)"
    @info "Relative reconstruction H1-error: $H1_err_nb (snapshot number $μ_nb)"

  end

  string_param_nbs = "params"
  for param_nb in param_nbs
    string_param_nbs *= "_" * string(param_nb)
  end

  if ROM_info.save_results

    path_μ = joinpath(ROM_info.paths.results_path, string_param_nbs)
    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV([mean_H1_err], joinpath(path_μ, "H1_err.csv"))

    if !ROM_info.import_offline_structures
      times = [RB_variables.offline_time, mean_online_time, mean_reconstruction_time]
    else
      times = [mean_online_time, mean_reconstruction_time]
    end
    save_CSV(times, joinpath(path_μ, "times.csv"))

  end

end
