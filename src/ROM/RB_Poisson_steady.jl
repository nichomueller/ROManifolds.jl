include("RB_utils.jl")
include("S-GRB_Poisson.jl")
include("S-PGRB_Poisson.jl")

function get_snapshot_matrix(ROM_info::Info, RB_variables::PoissonSteady)

  @info "Importing the snapshot matrix for field u, number of snapshots considered: $(ROM_info.nₛ)"

  name = "uₕ"
  Sᵘ = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, name * ".csv"), DataFrame))[:, 1:ROM_info.nₛ]

  @info "Dimension of snapshot matrix: $(size(Sᵘ))"

  RB_variables.Sᵘ = Sᵘ
  RB_variables.Nₛᵘ = size(Sᵘ)[1]

end

function get_norm_matrix(ROM_info::Info, RB_variables::PoissonSteady)

  if check_norm_matrix(RB_variables)

    @info "Importing the norm matrix Xᵘ₀"

    Xᵘ₀ = load_CSV(joinpath(ROM_info.paths.FEM_structures_path, "Xᵘ₀.csv"); convert_to_sparse = true)
    RB_variables.Nₛᵘ = size(Xᵘ₀)[1]
    @info "Dimension of norm matrix: $(size(Xᵘ₀))"

    if ROM_info.use_norm_X
      RB_variables.Xᵘ₀ = Xᵘ₀
    else
      RB_variables.Xᵘ₀ = I(RB_variables.Nₛᵘ)
    end

  end

end

function check_norm_matrix(RB_variables::PoissonSteady) :: Bool

  isempty(RB_variables.Xᵘ₀) || maximum(abs.(RB_variables.Xᵘ₀)) == 0

end

function PODs_space(ROM_info::Info, RB_variables::PoissonSteady)

  @info "Performing the spatial POD for field u, using a tolerance of $(ROM_info.ϵₛ)"

  get_norm_matrix(ROM_info, RB_variables)
  Φₛᵘ, _ = POD(RB_variables.Sᵘ, ROM_info.ϵₛ, RB_variables.Xᵘ₀)

  RB_variables.Φₛᵘ = Φₛᵘ
  (RB_variables.Nₛᵘ, RB_variables.nₛᵘ) = size(Φₛᵘ)

end

function build_reduced_basis(ROM_info::Info, RB_variables::PoissonSteady)

  @info "Building the spatial reduced basis for field u, using a tolerance of $(ROM_info.ϵₛ)"

  RB_building_time = @elapsed begin
    PODs_space(ROM_info, RB_variables)
  end

  RB_variables.offline_time += RB_building_time

  if ROM_info.save_offline_structures
    save_CSV(RB_variables.Φₛᵘ, joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
  end

end

function import_reduced_basis(ROM_info::Info, RB_variables::PoissonSteady)

  @info "Importing the spatial reduced basis for field u"
  RB_variables.Φₛᵘ = load_CSV(joinpath(ROM_info.paths.basis_path, "Φₛᵘ.csv"))
  (RB_variables.Nₛᵘ, RB_variables.nₛᵘ) = size(RB_variables.Φₛᵘ)

end

function get_generalized_coordinates(ROM_info::Info, RB_variables::PoissonSteady, snaps=nothing)

  get_norm_matrix(ROM_info, RB_variables)

  if isnothing(snaps) || maximum(snaps) > ROM_info.nₛ
    snaps = 1:ROM_info.nₛ
  end

  Φₛᵘ_normed = RB_variables.Xᵘ₀ * RB_variables.Φₛᵘ
  RB_variables.û = RB_variables.Sᵘ * Φₛᵘ_normed

  if ROM_info.save_offline_structures
    save_CSV(RB_variables.û, joinpath(ROM_info.paths.gen_coords_path, "û.csv"))
  end

end

function set_operators(ROM_info::Info, RB_variables::PoissonSteady) :: Vector

  operators = ["A"]
  if !ROM_info.build_parametric_RHS && !ROM_info.probl_nl["f"]
    append!(operators, ["F"])
  end
  if !ROM_info.build_parametric_RHS && !ROM_info.probl_nl["h"]
    append!(operators, ["H"])
  end

  operators

end

function save_M_DEIM_structures(ROM_info::Info, RB_variables::PoissonSteady)

  list_M_DEIM = (RB_variables.MDEIMᵢ_A, RB_variables.MDEIM_idx_A, RB_variables.sparse_el_A, RB_variables.DEIMᵢ_mat_F, RB_variables.DEIM_idx_F, RB_variables.DEIMᵢ_mat_H, RB_variables.DEIM_idx_H)
  list_names = ("MDEIMᵢ_A", "MDEIM_idx_A", "sparse_el_A", "DEIMᵢ_mat_F", "DEIM_idx_F", "DEIMᵢ_mat_H", "DEIM_idx_H")
  l_info_vec = [[l_idx,l_val] for (l_idx,l_val) in enumerate(list_M_DEIM) if !all(isempty.(l_val))]

  if !isempty(l_info_vec)
    l_info_mat = reduce(vcat,transpose.(l_info_vec))
    l_idx,l_val = l_info_mat[:,1], transpose.(l_info_mat[:,2])
    for (i₁,i₂) in enumerate(l_idx)
      save_CSV(l_val[i₁], joinpath(ROM_info.paths.ROM_structures_path, list_names[i₂]*".csv"))
    end
  end

end

function get_M_DEIM_structures(ROM_info::Info, RB_variables::PoissonSteady) :: Vector

  operators = String[]

  if ROM_info.probl_nl["A"]

    if isfile(joinpath(ROM_info.paths.ROM_structures_path, "MDEIMᵢ_A.csv"))
      @info "Importing MDEIM offline structures for the stiffness matrix"
      RB_variables.MDEIMᵢ_A = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIMᵢ_A.csv"))
      RB_variables.MDEIM_idx_A = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "MDEIM_idx_A.csv"))[:]
      RB_variables.sparse_el_A = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "sparse_el_A.csv"))[:]
      append!(operators, [])
    else
      @info "Failed to import MDEIM offline structures for the stiffness matrix: must build them"
      append!(operators, ["A"])
    end

  end

  if ROM_info.build_parametric_RHS

    @info "Will assemble nonaffine reduced RHS exactly"
    return operators

  else

    if ROM_info.probl_nl["f"]

      if isfile(joinpath(ROM_info.paths.ROM_structures_path, "DEIMᵢ_mat_F.csv"))
        @info "Importing DEIM offline structures for the forcing vector"
        RB_variables.DEIMᵢ_mat_F = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "DEIMᵢ_mat_F.csv"))
        RB_variables.DEIM_idx_F = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_idx_F.csv"))[:]
        append!(operators, [])
      else
        @info "Failed to import DEIM offline structures for the forcing vector: must build them"
        append!(operators, ["F"])
      end

    end

    if ROM_info.probl_nl["h"]

      if isfile(joinpath(ROM_info.paths.ROM_structures_path, "DEIMᵢ_mat_H.csv"))
        @info "Importing DEIM offline structures for the Neumann vector"
        RB_variables.DEIMᵢ_mat_H = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "DEIMᵢ_mat_H.csv"))
        RB_variables.DEIM_idx_H = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "DEIM_idx_H.csv"))[:]
        append!(operators, [])
        return
      else
        @info "Failed to import DEIM offline structures for the Neumann vector: must build them"
        append!(operators, ["H"])
      end

    end

  end

  operators

end

function get_Fₙ(ROM_info::Info, RB_variables::RBProblem) :: Vector

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
    @info "Importing reduced forcing vector"
    RB_variables.Fₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Fₙ.csv"))
    return []
  else
    @info "Failed to import the reduced forcing vector: must build it"
    return ["F"]
  end

end

function get_Hₙ(ROM_info::Info, RB_variables::RBProblem) :: Vector

  if isfile(joinpath(ROM_info.paths.ROM_structures_path, "Hₙ.csv"))
    @info "Importing reduced affine Neumann data vector"
    RB_variables.Hₙ = load_CSV(joinpath(ROM_info.paths.ROM_structures_path, "Hₙ.csv"))
    return []
  else
    @info "Failed to import the reduced affine Neumann data vector: must build it"
    return ["H"]
  end

end

function get_affine_structures(ROM_info::Info, RB_variables::PoissonSteady) :: Vector

  operators = String[]

  append!(operators, get_Aₙ(ROM_info, RB_variables))

  if ROM_info.build_parametric_RHS
    return operators
  else
    append!(operators, get_Fₙ(ROM_info, RB_variables))
    append!(operators, get_Hₙ(ROM_info, RB_variables))
  end

  operators

end

function get_offline_structures(ROM_info::Info, RB_variables::PoissonSteady) :: Vector

  operators = String[]

  append!(operators, get_affine_structures(ROM_info, RB_variables))
  append!(operators, get_M_DEIM_structures(ROM_info, RB_variables))
  unique!(operators)

  operators

end

function assemble_offline_structures(ROM_info::Info, RB_variables::PoissonSteady, operators=nothing)

  if isnothing(operators)
    operators = set_operators(ROM_info, RB_variables)
  end

  assembly_time = 0
  if "A" ∈ operators || "F" ∈ operators || "H" ∈ operators
    assembly_time += @elapsed begin
      if !ROM_info.probl_nl["A"]
        assemble_affine_matrices(ROM_info, RB_variables, "A")
      else
        assemble_MDEIM_matrices(ROM_info, RB_variables, "A")
      end
    end
  end

  if "F" ∈ operators
    assembly_time += @elapsed begin
      if !ROM_info.probl_nl["f"]
        assemble_affine_vectors(ROM_info, RB_variables, "F")
      else
        assemble_DEIM_vectors(ROM_info, RB_variables, "F")
      end
    end
  end

  if "H" ∈ operators
    assembly_time += @elapsed begin
      if !ROM_info.probl_nl["h"]
        assemble_affine_vectors(ROM_info, RB_variables, "H")
      else
        assemble_DEIM_vectors(ROM_info, RB_variables, "H")
      end
    end
  end
  RB_variables.offline_time += assembly_time

  save_affine_structures(ROM_info, RB_variables)
  save_M_DEIM_structures(ROM_info, RB_variables)

end

function get_system_blocks(ROM_info::Info, RB_variables::RBProblem, LHS_blocks::Array, RHS_blocks::Array) :: Vector

  if !ROM_info.import_offline_structures
    return ["LHS", "RHS"]
  end

  operators = String[]

  for i = LHS_blocks

    LHSₙi = "LHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(ROM_info.paths.ROM_structures_path, LHSₙi * ".csv"))
      append!(operators, ["LHS"])
      break
    end

  end

  for i = RHS_blocks

    RHSₙi = "RHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(ROM_info.paths.ROM_structures_path, RHSₙi * ".csv"))
      append!(operators, ["RHS"])
      break
    end

  end

  if "LHS" ∉ operators

    for i = LHS_blocks

      LHSₙi = "LHSₙ" * string(i) * ".csv"
      @info "Importing block number $i of the reduced affine LHS"
      push!(RB_variables.LHSₙ, load_CSV(joinpath(ROM_info.paths.ROM_structures_path, LHSₙi)))
      RB_variables.nᵘ = size(RB_variables.LHSₙ[i])[1]

    end

  end

  if "RHS" ∉ operators

    for i = RHS_blocks

      RHSₙi = "RHSₙ" * string(i) * ".csv"
      @info "Importing block number $i of the reduced affine LHS"
      push!(RB_variables.RHSₙ, load_CSV(joinpath(ROM_info.paths.ROM_structures_path, RHSₙi)))
      RB_variables.nᵘ = size(RB_variables.RHSₙ[i])[1]

    end

  end

  operators

end

function get_θᵃ(ROM_info::Info, RB_variables::PoissonSteady, param) :: Array

  if !ROM_info.probl_nl["A"]
    θᵃ = param.α(Point(0.,0.))
  else
    A_μ_sparse = build_sparse_mat(problem_info, FE_space, param, RB_variables.sparse_el_A)
    θᵃ = M_DEIM_online(A_μ_sparse, RB_variables.MDEIMᵢ_A, RB_variables.MDEIM_idx_A)
  end

  return θᵃ

end

function get_θᶠʰ(ROM_info::Info, RB_variables::PoissonSteady, param) :: Tuple

  if ROM_info.build_parametric_RHS
    @error "Cannot fetch θᶠ, θʰ if the RHS is built online"
  end

  if !ROM_info.probl_nl["f"] && !ROM_info.probl_nl["h"]
    θᶠ, θʰ = param.f(Point(0.,0.)), param.h(Point(0.,0.))
  elseif !ROM_info.probl_nl["f"] && ROM_info.probl_nl["h"]
    H_μ = assemble_neumann_datum(FE_space, ROM_info, param)
    θᶠ, θʰ = param.f(Point(0.,0.)), M_DEIM_online(H_μ, RB_variables.DEIMᵢ_mat_H, RB_variables.DEIM_idx_H)
  elseif ROM_info.probl_nl["f"] && !ROM_info.probl_nl["h"]
    F_μ = assemble_forcing(FE_space, ROM_info, param)
    θᶠ, θʰ = M_DEIM_online(F_μ, RB_variables.DEIMᵢ_mat_F, RB_variables.DEIM_idx_F), param.h(Point(0.,0.))
  else ROM_info.probl_nl["f"] && ROM_info.probl_nl["h"]
    F_μ = assemble_forcing(FE_space, ROM_info, param)
    H_μ = assemble_neumann_datum(FE_space, ROM_info, param)
    θᶠ, θʰ = M_DEIM_online(F_μ, RB_variables.DEIMᵢ_mat_F, RB_variables.DEIM_idx_F), M_DEIM_online(H_μ, RB_variables.DEIMᵢ_mat_H, RB_variables.DEIM_idx_H)
  end

  return θᶠ, θʰ

end

function initialize_RB_system(RB_variables::RBProblem)

  RB_variables.LHSₙ = Matrix{Float64}[]
  RB_variables.RHSₙ = Matrix{Float64}[]

end

function get_Q(ROM_info::Info, RB_variables::PoissonSteady)

  if RB_variables.Qᵃ == 0
    RB_variables.Qᵃ = size(RB_variables.Aₙ)[end]
  end

  if !ROM_info.build_parametric_RHS
    if RB_variables.Qᶠ == 0
      RB_variables.Qᶠ = size(RB_variables.Fₙ)[end]
    end
    if RB_variables.Qʰ == 0
      RB_variables.Qʰ = size(RB_variables.Hₙ)[end]
    end
  end

end

function get_RB_system(ROM_info::Info, RB_variables::PoissonSteady, param)

  initialize_RB_system(RB_variables)
  get_Q(ROM_info, RB_variables)
  blocks = [1]
  operators = get_system_blocks(ROM_info, RB_variables, blocks, blocks)

  @info "Preparing the RB system: fetching reduced LHS"
  θᵃ, θᶠ, θʰ = get_θ(ROM_info, RB_variables, param)

  if "LHS" ∈ operators
    Aₙ_μ = assemble_online_structure(θᵃ, RB_variables.Aₙ)
    push!(RB_variables.LHSₙ, Aₙ_μ)
  end

  if "RHS" ∈ operators
    if !ROM_info.build_parametric_RHS
      @info "Preparing the RB system: fetching reduced RHS"
      Fₙ_μ = assemble_online_structure(θᶠ, RB_variables.Fₙ)
      Hₙ_μ = assemble_online_structure(θʰ, RB_variables.Hₙ)
      push!(RB_variables.RHSₙ, reshape(Fₙ_μ+Hₙ_μ,:,1))
    else
      @info "Preparing the RB system: assembling reduced RHS exactly"
      Fₙ_μ, Hₙ_μ = build_param_RHS(ROM_info, RB_variables, param, θᵃ)
      push!(RB_variables.RHSₙ, reshape(Fₙ_μ+Hₙ_μ,:,1))
    end
  end

end

function solve_RB_system(ROM_info::Info, RB_variables::PoissonSteady, param)

  get_RB_system(ROM_info, RB_variables, param)

  @info "Solving RB problem via backslash"
  @info "Condition number of the system's matrix: $(cond(RB_variables.LHSₙ[1]))"
  RB_variables.uₙ = zeros(RB_variables.nₛᵘ)
  RB_variables.uₙ = RB_variables.LHSₙ[1] \ RB_variables.RHSₙ[1]

end

function reconstruct_FEM_solution(RB_variables::PoissonSteady)


  @info "Reconstructing FEM solution from the newly computed RB one"

  RB_variables.ũ = RB_variables.Φₛᵘ * RB_variables.uₙ

end

function build_RB_approximation(ROM_info::Info, RB_variables::PoissonSteady)


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
    operators = get_offline_structures(ROM_info, RB_variables)
    if "A" ∈ operators || "F" ∈ operators || "H" ∈ operators
      assemble_offline_structures(ROM_info, RB_variables, operators)
    end
  else
    assemble_offline_structures(ROM_info, RB_variables)
  end

end

function testing_phase(ROM_info::Info, RB_variables::PoissonSteady, μ, param_nbs)

  mean_H1_err = 0.0
  mean_pointwise_err = zeros(RB_variables.Nₛᵘ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(ROM_info, RB_variables)

  ũ_μ = zeros(RB_variables.Nₛᵘ, length(param_nbs))
  uₙ_μ = zeros(RB_variables.nₛᵘ, length(param_nbs))

  for nb in param_nbs
    @info "Considering parameter number: $nb"

    μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
    parametric_info = get_parametric_specifics(problem_ntuple, ROM_info, μ_nb)

    uₕ_test = Matrix(CSV.read(joinpath(ROM_info.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, nb]

    online_time = @elapsed begin
      solve_RB_system(ROM_info, RB_variables, parametric_info)
    end
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RB_variables)
    end
    mean_online_time = online_time / length(param_nbs)
    mean_reconstruction_time = reconstruction_time / length(param_nbs)

    H1_err_nb = compute_errors(uₕ_test, RB_variables, RB_variables.Xᵘ₀)
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_pointwise_err += abs.(uₕ_test - RB_variables.ũ) / length(param_nbs)

    ũ_μ[:, nb - param_nbs[1] + 1] = RB_variables.ũ
    uₙ_μ[:, nb - param_nbs[1] + 1] = RB_variables.uₙ

    @info "Online wall time: $online_time s (snapshot number $nb)"
    @info "Relative reconstruction H1-error: $H1_err_nb (snapshot number $nb)"

  end

  string_param_nbs = "params"
  for param_nb in param_nbs
    string_param_nbs *= "_" * string(param_nb)
  end
  path_μ = joinpath(ROM_info.paths.results_path, string_param_nbs)

  if ROM_info.save_results

    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV([mean_H1_err], joinpath(path_μ, "H1_err.csv"))

    if !ROM_info.import_offline_structures
      times = [RB_variables.offline_time, mean_online_time, mean_reconstruction_time]
    else
      times = [mean_online_time, mean_reconstruction_time]
    end
    save_CSV(times, joinpath(path_μ, "times.csv"))

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FE_space"=>FE_space, "mean_point_err_u"=>mean_pointwise_err)

  if ROM_info.postprocess
    post_process(ROM_info, pass_to_pp)
  end

end
