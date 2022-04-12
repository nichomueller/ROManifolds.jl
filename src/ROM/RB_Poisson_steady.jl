include("../utils/general.jl")
include("RB_utils.jl")
include("../FEM/FEM.jl")

function get_snapshot_matrix(ROM_info, RB_variables::RBProblem)
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


function get_norm_matrix(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  if check_norm_matrix(RB_variables)

    @info "Importing the norm matrix"

    Xᵘ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Xᵘ.csv"); convert_to_sparse = true)
    RB_variables.Xᵘ = Xᵘ
    RB_variables.Nₛᵘ = size(Xᵘ)[1]
    #= try
      Xᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_structures_path, "Xᵘ.csv"), DataFrame))
      RB_variables.Xᵘ = Xᵘ
    catch e
      println("Error: $e. Impossible to load the H1 norm matrix")
    end =#

    @info "Dimension of norm matrix: $(size(RB_variables.Xᵘ))"

  end

end


function check_norm_matrix(RB_variables::RBProblem)
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

function PODs_space(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  @info "Performing the spatial POD for field u, using a tolerance of $(ROM_info.ϵₛ)"

  get_norm_matrix(ROM_info, RB_variables)
  Φₛᵘ, _ = POD(RB_variables.Sᵘ, ROM_info.ϵₛ)
  #Φₛᵘ, _ = POD(RB_variables.Sᵘ, ROM_info.ϵₛ, RB_variables.Xᵘ) #SOMETHING WRONG HERE...FIX NORM MATRIX!

  RB_variables.Φₛᵘ = Φₛᵘ
  (RB_variables.Nₛᵘ, RB_variables.nₛᵘ) = size(Φₛᵘ)

end


function build_reduced_basis(ROM_info, RB_variables::RBProblem)
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


function import_reduced_basis(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  @info "Importing the reduced basis"

  RB_variables.Φₛᵘ = load_CSV(joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
  (RB_variables.Nₛᵘ, RB_variables.nₛᵘ) = size(RB_variables.Φₛᵘ)

end


function check_reduced_affine_components(ROM_info, RB_variables::RBProblem)
  #=MODIFY
  =#

  operators = []

  if !ROM_info.problem_nonlinearities["A"]

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
      @info "Importing reduced affine stiffness matrix"
      RB_variables.Aₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
    else
      @info "Failed to import the reduced affine stiffness matrix: must build it"
      push!(operators, "A")
    end

  else

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_affine.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Aₙ_idx.csv"))
      @info "Importing MDEIM offline structures for the stiffness matrix"
      RB_variables.MDEIM_mat = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_mat.csv"))
      RB_variables.MDEIM_idx = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_idx.csv"))[:]
      RB_variables.row_idx = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "row_idx.csv"))[:]
      RB_variables.col_idx = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "col_idx.csv"))[:]
      RB_variables.sparse_el = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "sparse_el.csv"))[:]
    else
      @info "Failed to import MDEIM offline structures for the stiffness matrix: must build them"
      push!(operators, "A")
    end

  end

  if !ROM_info.problem_nonlinearities["f"] && !ROM_info.problem_nonlinearities["h"]

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
      @info "Importing reduced affine RHS vector"
      RB_variables.Fₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
    else
      @info "Failed to import the reduced affine RHS vector: must build it"
      push!(operators, "F")
    end

  else

    if ROM_info.perform_RHS_DEIM

      if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_affine.csv")) && isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_idx.csv"))
        @info "Importing DEIM offline structures for the RHS vector"
        RB_variables.Fₙ_affine = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_affine.csv"))
        RB_variables.Fₙ_idx = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_idx.csv"))
      else
        @info "Failed to import DEIM offline structures for the RHS vector: must build them"
        push!(operators, "F")
      end

    else

      @info "Will assemble nonaffine reduced RHS exactly"

    end

  end

  operators

end

function get_generalized_coordinates(ROM_info, RB_variables::RBProblem, snaps=nothing)
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

function set_operators(ROM_info)

  operators = ["A"]
  if ROM_info.perform_RHS_DEIM || (!ROM_info.problem_nonlinearities["f"] && !ROM_info.problem_nonlinearities["h"])
    push!(operators, "F")
  end

  operators

end

function assemble_reduced_affine_components(ROM_info, RB_variables::RBProblem, operators=nothing; μ=nothing)
  #=MODIFY
  =#

  if operators === nothing
    operators = set_operators(ROM_info)
  end

  projection_time = 0

  if "A" in operators

    if ROM_info.problem_nonlinearities["A"] === false

      @info "Assembling affine reduced stiffness"
      projection_time += @elapsed begin
        A = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "A.csv"); convert_to_sparse = true)
        RB_variables.Aₙ = (RB_variables.Φₛᵘ)' * A * RB_variables.Φₛᵘ
        if ROM_info.save_offline_structures
          save_CSV(RB_variables.Aₙ, joinpath(ROM_info.paths.ROM_structures_path, "Aₙ.csv"))
        end
      end

    else

      nₛ_MDEIM = max(35, ROM_info.nₛ)
      @info "The stiffness is non-affine: running the MDEIM offline phase on $nₛ_MDEIM snapshots"

      projection_time += @elapsed begin

        RB_variables.MDEIM_mat, RB_variables.MDEIM_idx, RB_variables.row_idx, RB_variables.col_idx, RB_variables.sparse_el, MDEIM_err_bound, Σ = MDEIM_offline(problem_info, ROM_info, nₛ_MDEIM, μ)

      end

      if ROM_info.save_offline_structures
        save_CSV(RB_variables.MDEIM_mat, joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_mat.csv"))
        save_CSV(RB_variables.MDEIM_idx, joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_idx.csv"))
        save_CSV(RB_variables.row_idx, joinpath(ROM_info.paths.ROM_structures_path, "row_idx.csv"))
        save_CSV(RB_variables.col_idx, joinpath(ROM_info.paths.ROM_structures_path, "col_idx.csv"))
        save_CSV(RB_variables.sparse_el, joinpath(ROM_info.paths.ROM_structures_path, "sparse_el.csv"))
        save_CSV([MDEIM_err_bound], joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_err_bound.csv"))
        save_CSV(Σ, joinpath(ROM_info.paths.ROM_structures_path, "Σ.csv"))
      end

    end

  end

  if "F" in operators

    if !ROM_info.problem_nonlinearities["f"] && !ROM_info.problem_nonlinearities["h"]

      @info "Assembling affine reduced forcing term"
      projection_time += @elapsed begin
        F = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "F.csv"))
        RB_variables.Fₙ = (RB_variables.Φₛᵘ)' * F
        if ROM_info.save_offline_structures
          save_CSV(RB_variables.Fₙ, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
        end
      end

    else

      nₛ_DEIM = min(50, ROM_info.nₛ)
      @info "The forcing term is non-affine: running the DEIM offline phase on $nₛ_DEIM snapshots"
      Fₙ_i = zeros(RB_variables.nₛᵘ, nₛ_DEIM)
      projection_time += @elapsed begin
        for i_nₛ = 1:nₛ_DEIM
          μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
          parametric_info = get_parametric_specifics(ROM_info, μ_i)
          FE_space = get_FE_space(problem_info, parametric_info.model)
          F_i = assemble_forcing(FE_space, parametric_info)
          Fₙ_i[:, i_nₛ] = (RB_variables.Φₛᵘ)' * F_i
        end

        (RB_variables.Fₙ_affine, RB_variables.Fₙ_idx) = DEIM_offline(Fₙ_i, ROM_info.ϵₛ ^ 2)
        if ROM_info.save_offline_structures
          save_CSV(RB_variables.Fₙ_affine, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_affine.csv"))
          save_CSV(RB_variables.Fₙ_idx, joinpath(ROM_info.paths.ROM_structures_path, "Fₙ_idx.csv"))
        end

      end
    end

  end

  RB_variables.offline_time += projection_time

end

function build_parametric_RHS(ROM_info, RB_variables::PoissonSTGRB, μ::Array)

  projection_time = 0
  projection_time += @elapsed begin
    parametric_info = get_parametric_specifics(ROM_info, μ)
    FE_space = get_FE_space(problem_info, parametric_info.model)
    F = assemble_forcing(FE_space, parametric_info)
    Fₙ = (RB_variables.Φₛᵘ)' * F
  end

  Fₙ

end

function build_parametric_RHS(ROM_info, RB_variables::PoissonSTPGRB, μ::Array)

  projection_time = 0
  projection_time += @elapsed begin
    parametric_info = get_parametric_specifics(ROM_info, μ)
    FE_space = get_FE_space(problem_info, parametric_info.model)
    A = assemble_stiffness(FE_space, ROM_info, parametric_info)
    F = assemble_forcing(FE_space, parametric_info)
    AΦₛᵘ = A * RB_variables.Φₛᵘ
    Fₙ = (AΦₛᵘ)' * RB_variables.Pᵘ_inv * F
  end

  Fₙ

end

function initialize_RB_system(RB_variables::RBProblem)
  #=MODIFY
  =#

  RB_variables.LHSₙ = Matrix{Float64}[]
  RB_variables.RHSₙ = Matrix{Float64}[]

end

function get_RB_system(ROM_info, RB_variables::RBProblem, param; FE_space = nothing)
  #=MODIFY
  =#

  @info "Preparing the RB system: fetching online reduced structures"

  initialize_RB_system(RB_variables)

  if ROM_info.problem_nonlinearities["A"] === false
    push!(RB_variables.LHSₙ, param.α(Point(0.,0.)) * RB_variables.Aₙ)
  else
    A_μ_sparse = build_sparse_LHS(problem_info, ROM_info, param.μ, RB_variables.sparse_el)
    _, A_μ_affine_sparse = MDEIM_online(A_μ_sparse, RB_variables.MDEIM_mat, RB_variables.MDEIM_idx, RB_variables.row_idx, RB_variables.col_idx)
    Aₙ_μ_affine = (RB_variables.Φₛᵘ)' * A_μ_affine_sparse * RB_variables.Φₛᵘ
    push!(RB_variables.LHSₙ, Aₙ_μ_affine)
  end

  if ROM_info.problem_nonlinearities["f"] === false && ROM_info.problem_nonlinearities["h"] === false
    push!(RB_variables.RHSₙ, RB_variables.Fₙ)
  else
    if ROM_info.perform_RHS_DEIM
      Fₙ_μ = (RB_variables.Φₛᵘ)' * assemble_forcing(FE_space, param)
      _, Fₙ_μ_affine = DEIM_online(Fₙ_μ, RB_variables.Fₙ_affine, RB_variables.Fₙ_idx)
      push!(RB_variables.RHSₙ, reshape(Fₙ_μ_affine, :, 1))
    else
      @info "Assembling nonaffine reduced RHS exactly"
      Fₙ_μ = build_parametric_RHS(ROM_info, RB_variables, param.μ)
      push!(RB_variables.RHSₙ, reshape(Fₙ_μ, :, 1))
    end
  end

end

function solve_RB_system(ROM_info, RB_variables::RBProblem, param; FE_space=nothing)
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

function reconstruct_FEM_solution(RB_variables::RBProblem)
  #=MODIFY
  =#

  @info "Reconstructing FEM solution from the newly computed RB one"

  RB_variables.ũ = zeros(RB_variables.Nₛᵘ)
  RB_variables.ũ = RB_variables.Φₛᵘ * RB_variables.uₙ

end

function build_RB_approximation(ROM_info, RB_variables::RBProblem; μ=nothing)
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


function testing_phase(ROM_info, RB_variables::RBProblem, μ, param_nbs)
  #=MODIFY
  =#

  mean_H1_err = 0.0
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(ROM_info, RB_variables)

  ũ_μ = zeros(RB_variables.Nₛᵘ, length(param_nbs))
  uₙ_μ = zeros(RB_variables.nₛᵘ, length(param_nbs))
  FE_space = nothing

  for nb in param_nbs
    @info "Considering parameter number: $nb"

    μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_nb)
    if ROM_info.case > 0
      FE_space = get_FE_space(problem_info, parametric_info.model)
    end

    uₕ_test = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, nb]

    #= proj_err = uₕ_test - RB_variables.Φₛᵘ * RB_variables.Φₛᵘ' * uₕ_test
    generate_vtk_file(FE_space, ROM_info.paths.results_path, "err", proj_err) =#
    #= function check_offline_phase()
      A = assemble_stiffness(FE_space, problem_info, parametric_info)
      Aₙ = RB_variables.Φₛᵘ' * A * RB_variables.Φₛᵘ
      F = assemble_forcing(FE_space, parametric_info)
      Fₙ = RB_variables.Φₛᵘ' * F
      uₙ = Aₙ \ Fₙ
      uₕ_approx = RB_variables.Φₛᵘ * uₙ
      compute_errors(uₕ_test, uₕ_approx)
    end

    function check_mdeim()
      A = assemble_stiffness(FE_space, problem_info, parametric_info)
      A_μ_sparse = build_sparse_LHS(problem_info, ROM_info, μ_nb, RB_variables.sparse_el)
      _, A_μ_affine_sparse = MDEIM_online(A_μ_sparse, RB_variables.MDEIM_mat, RB_variables.MDEIM_idx, RB_variables.row_idx, RB_variables.col_idx)
      norm(A - A_μ_affine_sparse)
    end =#

    online_time = @elapsed begin
      solve_RB_system(ROM_info, RB_variables, parametric_info; FE_space = FE_space)
    end
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RB_variables)
    end
    mean_online_time = online_time / length(param_nbs)
    mean_reconstruction_time = reconstruction_time / length(param_nbs)

    H1_err_nb = compute_errors(uₕ_test, RB_variables.ũ)#, RB_variables.Xᵘ)
    mean_H1_err += H1_err_nb / length(param_nbs)

    ũ_μ[:, nb - param_nbs[1] + 1] = RB_variables.ũ
    uₙ_μ[:, nb - param_nbs[1] + 1] = RB_variables.uₙ

    @info "Online wall time: $online_time s (snapshot number $nb)"
    @info "Relative reconstruction H1-error: $H1_err_nb (snapshot number $nb)"

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

  if ROM_info.postprocess
    post_process()
  end

end

function post_process()

end


#= ### test MDEIMs
  μ_nb = parse.(Float64, split(chop(μ[1]; head=1, tail=1), ','))
  parametric_info = get_parametric_specifics(ROM_info, μ_nb)
  FE_space = get_FE_space(problem_info, parametric_info.model)
  nₛ_MDEIM = 20
  err_l2, on_time_MDEIM = test_MDEIM_Manzoni(ROM_info, RB_variables, FE_space, nₛ_MDEIM, μ, param_nbs)
  my_err_l2, my_on_time_MDEIM = test_my_MDEIM(ROM_info, RB_variables, nₛ_MDEIM, μ, param_nbs)
  ### end test =#

#= function test_MDEIM_Manzoni(ROM_info, RB_variables, FE_space, nₛ_MDEIM, μ, param_nbs)
  #= S_sparse, row_idx, col_idx = build_A_snapshots(problem_info, ROM_info, RB_variables, nₛ_MDEIM, μ)
  U_sparse = MPOD(S_sparse, ROM_info.ϵₛ)
  I_sparse = MDEIM(U_sparse)

  row_idx = row_idx[I_sparse]
  col_idx = col_idx[I_sparse]
  idx = union(row_idx, col_idx)
  el = Int64[]
  for i = 1:length(idx)
    for j = 1:size(FE_space.σₖ)[1]
      if el[i] in FE_space.σₖ[j, :]
        append!(el, j)
      end
    end
  end =#

  S, row_idx, col_idx = build_A_snapshots(problem_info, ROM_info, nₛ_MDEIM, μ)
  DEIM_mat, DEIM_idx = DEIM_offline(S, ROM_info.ϵₛ)
  idx = union(row_idx[DEIM_idx], col_idx[DEIM_idx])

  sparse_el = find_FE_elements(idx, FE_space.σₖ)

  err = sparse(zeros(RB_variables.Nₛᵘ, RB_variables.Nₛᵘ))
  time_MDEIM_on = @elapsed begin
    for i in param_nbs
      μ_i = parse.(Float64, split(chop(μ[i]; head=1, tail=1), ','))
      parametric_info = get_parametric_specifics(ROM_info, μ_i)
      FE_space = get_FE_space(problem_info, parametric_info.model)
      A = assemble_stiffness(FE_space, problem_info, parametric_info)
      A_sparse = build_sparse_LHS(problem_info, ROM_info, μ_i)
      _, A_affine_sparse = MDEIM_online(A_sparse, DEIM_mat, DEIM_idx, row_idx, col_idx)
      err += abs.(A - A_affine_sparse)
    end
  end

  norm(err) / length(param_nbs), time_MDEIM_on

end

function test_my_MDEIM(ROM_info, RB_variables, nₛ_MDEIM, μ, param_nbs)

  Aₙ_i = zeros(RB_variables.nₛᵘ ^ 2, nₛ_MDEIM)
  for i_nₛ = 1:nₛ_MDEIM
    μ_i = parse.(Float64, split(chop(μ[i_nₛ]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(ROM_info, μ_i)
    FE_space = get_FE_space(problem_info, parametric_info.model)
    A_i = assemble_stiffness(FE_space, ROM_info, parametric_info)
    Aₙ_i[:, i_nₛ] = reshape((RB_variables.Φₛᵘ)' * A_i * RB_variables.Φₛᵘ, :, 1)
  end

  (Aₙ_affine, Aₙ_idx) = DEIM_offline(Aₙ_i, ROM_info.ϵₛ)

  err = sparse(zeros(RB_variables.Nₛᵘ, RB_variables.Nₛᵘ))
  time_MDEIM_on = @elapsed begin
    for i in param_nbs
      μ_i = parse.(Float64, split(chop(μ[i]; head=1, tail=1), ','))
      parametric_info = get_parametric_specifics(ROM_info, μ_i)
      FE_space = get_FE_space(problem_info, parametric_info.model)
      A = assemble_stiffness(FE_space, problem_info, parametric_info)
      Aₙ = (RB_variables.Φₛᵘ)' * A * RB_variables.Φₛᵘ
      _, Aₙ_on = MDEIM_online(Aₙ, Aₙ_affine, Aₙ_idx)
      err += abs.(A - RB_variables.Φₛᵘ * Aₙ_on * (RB_variables.Φₛᵘ)')
    end
  end

  norm(err), time_MDEIM_on

end =#
