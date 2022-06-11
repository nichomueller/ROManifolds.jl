include("RB.jl")
include("S-GRB_Poisson.jl")
include("S-PGRB_Poisson.jl")

function get_snapshot_matrix(RBInfo::Info, RBVars::PoissonSteady)

  @info "Importing the snapshot matrix for field u, number of snapshots considered: $(RBInfo.nₛ)"

  name = "uₕ"
  Sᵘ = (Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path,
  name * ".csv"), DataFrame))[:, 1:RBInfo.nₛ])

  @info "Dimension of snapshot matrix: $(size(Sᵘ))"

  RBVars.Sᵘ = Sᵘ
  RBVars.Nₛᵘ = size(Sᵘ)[1]

end

function get_norm_matrix(RBInfo::Info, RBVars::PoissonSteady)

  if check_norm_matrix(RBVars)

    @info "Importing the norm matrix Xᵘ₀"

    Xᵘ₀ = load_CSV(joinpath(RBInfo.paths.FEM_structures_path, "Xᵘ₀.csv");
     convert_to_sparse = true)
    RBVars.Nₛᵘ = size(Xᵘ₀)[1]
    @info "Dimension of norm matrix: $(size(Xᵘ₀))"

    if RBInfo.use_norm_X
      RBVars.Xᵘ₀ = Xᵘ₀
    else
      RBVars.Xᵘ₀ = I(RBVars.Nₛᵘ)
    end

  end

end

function check_norm_matrix(RBVars::PoissonSteady) :: Bool

  isempty(RBVars.Xᵘ₀) || maximum(abs.(RBVars.Xᵘ₀)) == 0

end

function PODs_space(RBInfo::Info, RBVars::PoissonSteady)

  @info "Performing the spatial POD for field u, using a tolerance of $(RBInfo.ϵₛ)"

  get_norm_matrix(RBInfo, RBVars)
  Φₛᵘ, _ = POD(RBVars.Sᵘ, RBInfo.ϵₛ, RBVars.Xᵘ₀)

  RBVars.Φₛᵘ = Φₛᵘ
  (RBVars.Nₛᵘ, RBVars.nₛᵘ) = size(Φₛᵘ)

end

function build_reduced_basis(RBInfo::Info, RBVars::PoissonSteady)

  @info "Building the spatial reduced basis for field u, using a tolerance of $(RBInfo.ϵₛ)"

  RB_building_time = @elapsed begin
    PODs_space(RBInfo, RBVars)
  end

  RBVars.offline_time += RB_building_time

  if RBInfo.save_offline_structures
    save_CSV(RBVars.Φₛᵘ, joinpath(RBInfo.paths.basis_path, "Φₛᵘ.csv"))
  end

end

function import_reduced_basis(RBInfo::Info, RBVars::PoissonSteady)

  @info "Importing the spatial reduced basis for field u"
  RBVars.Φₛᵘ = load_CSV(joinpath(RBInfo.paths.basis_path, "Φₛᵘ.csv"))
  (RBVars.Nₛᵘ, RBVars.nₛᵘ) = size(RBVars.Φₛᵘ)

end

function get_generalized_coordinates(
  RBInfo::Info,
  RBVars::PoissonSteady,
  snaps=nothing)

  get_norm_matrix(RBInfo, RBVars)

  if isnothing(snaps) || maximum(snaps) > RBInfo.nₛ
    snaps = 1:RBInfo.nₛ
  end

  Φₛᵘ_normed = RBVars.Xᵘ₀ * RBVars.Φₛᵘ
  RBVars.û = RBVars.Sᵘ * Φₛᵘ_normed

  if RBInfo.save_offline_structures
    save_CSV(RBVars.û, joinpath(RBInfo.paths.gen_coords_path, "û.csv"))
  end

end

function set_operators(RBInfo::Info, ::PoissonSteady) ::Vector

  operators = ["A"]
  if !RBInfo.build_Parametric_RHS
    append!(operators, ["F","H"])
  end

  operators

end

function save_M_DEIM_structures(RBInfo::Info, RBVars::PoissonSteady)

  list_M_DEIM = (RBVars.MDEIMᵢ_A, RBVars.MDEIM_idx_A, RBVars.sparse_el_A, RBVars.DEIMᵢ_mat_F, RBVars.DEIM_idx_F, RBVars.DEIMᵢ_mat_H, RBVars.DEIM_idx_H)
  list_names = ("MDEIMᵢ_A", "MDEIM_idx_A", "sparse_el_A", "DEIMᵢ_mat_F", "DEIM_idx_F", "DEIMᵢ_mat_H", "DEIM_idx_H")
  l_info_vec = [[l_idx,l_val] for (l_idx,l_val) in enumerate(list_M_DEIM) if !all(isempty.(l_val))]

  if !isempty(l_info_vec)
    l_info_mat = reduce(vcat,transpose.(l_info_vec))
    l_idx,l_val = l_info_mat[:,1], transpose.(l_info_mat[:,2])
    for (i₁,i₂) in enumerate(l_idx)
      save_CSV(l_val[i₁], joinpath(RBInfo.paths.ROM_structures_path, list_names[i₂]*".csv"))
    end
  end

end

function get_M_DEIM_structures(RBInfo::Info, RBVars::PoissonSteady) ::Vector

  operators = String[]

  if RBInfo.probl_nl["A"]

    if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MDEIMᵢ_A.csv"))
      @info "Importing MDEIM offline structures for the stiffness matrix"
      RBVars.MDEIMᵢ_A = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MDEIMᵢ_A.csv"))
      RBVars.MDEIM_idx_A = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_idx_A.csv"))[:]
      RBVars.sparse_el_A = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "sparse_el_A.csv"))[:]
      append!(operators, [])
    else
      @info "Failed to import MDEIM offline structures for the stiffness matrix: must build them"
      append!(operators, ["A"])
    end

  end

  if RBInfo.build_Parametric_RHS

    @info "Will assemble nonaffine reduced RHS exactly"
    return operators

  else

    if RBInfo.probl_nl["f"]

      if isfile(joinpath(RBInfo.paths.ROM_structures_path, "DEIMᵢ_mat_F.csv"))
        @info "Importing DEIM offline structures for the forcing vector"
        RBVars.DEIMᵢ_mat_F = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "DEIMᵢ_mat_F.csv"))
        RBVars.DEIM_idx_F = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_idx_F.csv"))[:]
        append!(operators, [])
      else
        @info "Failed to import DEIM offline structures for the forcing vector: must build them"
        append!(operators, ["F"])
      end

    end

    if RBInfo.probl_nl["h"]

      if isfile(joinpath(RBInfo.paths.ROM_structures_path, "DEIMᵢ_mat_H.csv"))
        @info "Importing DEIM offline structures for the Neumann vector"
        RBVars.DEIMᵢ_mat_H = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "DEIMᵢ_mat_H.csv"))
        RBVars.DEIM_idx_H = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_idx_H.csv"))[:]
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

function get_Fₙ(RBInfo::Info, RBVars::RBProblem) :: Vector

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Fₙ.csv"))
    @info "Importing reduced forcing vector"
    RBVars.Fₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Fₙ.csv"))
    return []
  else
    @info "Failed to import the reduced forcing vector: must build it"
    return ["F"]
  end

end

function get_Hₙ(RBInfo::Info, RBVars::RBProblem) :: Vector

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Hₙ.csv"))
    @info "Importing reduced affine Neumann data vector"
    RBVars.Hₙ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "Hₙ.csv"))
    return []
  else
    @info "Failed to import the reduced affine Neumann data vector: must build it"
    return ["H"]
  end

end

function get_affine_structures(RBInfo::Info, RBVars::PoissonSteady) :: Vector

  operators = String[]

  append!(operators, get_Aₙ(RBInfo, RBVars))

  if RBInfo.build_Parametric_RHS
    return operators
  else
    append!(operators, get_Fₙ(RBInfo, RBVars))
    append!(operators, get_Hₙ(RBInfo, RBVars))
  end

  operators

end

function get_offline_structures(RBInfo::Info, RBVars::PoissonSteady) :: Vector

  operators = String[]

  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function assemble_offline_structures(RBInfo::Info, RBVars::PoissonSteady, operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assembly_time = 0
  if "A" ∈ operators || "F" ∈ operators || "H" ∈ operators
    assembly_time += @elapsed begin
      if !RBInfo.probl_nl["A"]
        assemble_affine_matrices(RBInfo, RBVars, "A")
      else
        assemble_MDEIM_matrices(RBInfo, RBVars, "A")
      end
    end
  end

  if "F" ∈ operators
    assembly_time += @elapsed begin
      if !RBInfo.probl_nl["f"]
        assemble_affine_vectors(RBInfo, RBVars, "F")
      else
        assemble_DEIM_vectors(RBInfo, RBVars, "F")
      end
    end
  end

  if "H" ∈ operators
    assembly_time += @elapsed begin
      if !RBInfo.probl_nl["h"]
        assemble_affine_vectors(RBInfo, RBVars, "H")
      else
        assemble_DEIM_vectors(RBInfo, RBVars, "H")
      end
    end
  end
  RBVars.offline_time += assembly_time

  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function get_system_blocks(RBInfo::Info, RBVars::RBProblem, LHS_blocks::Array, RHS_blocks::Array) :: Vector

  if !RBInfo.import_offline_structures
    return ["LHS", "RHS"]
  end

  operators = String[]

  for i = LHS_blocks

    LHSₙi = "LHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(RBInfo.paths.ROM_structures_path, LHSₙi * ".csv"))
      append!(operators, ["LHS"])
      break
    end

  end

  for i = RHS_blocks

    RHSₙi = "RHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(RBInfo.paths.ROM_structures_path, RHSₙi * ".csv"))
      append!(operators, ["RHS"])
      break
    end

  end

  if "LHS" ∉ operators

    for i = LHS_blocks

      LHSₙi = "LHSₙ" * string(i) * ".csv"
      @info "Importing block number $i of the reduced affine LHS"
      push!(RBVars.LHSₙ, load_CSV(joinpath(RBInfo.paths.ROM_structures_path, LHSₙi)))
      RBVars.nᵘ = size(RBVars.LHSₙ[i])[1]

    end

  end

  if "RHS" ∉ operators

    for i = RHS_blocks

      RHSₙi = "RHSₙ" * string(i) * ".csv"
      @info "Importing block number $i of the reduced affine LHS"
      push!(RBVars.RHSₙ, load_CSV(joinpath(RBInfo.paths.ROM_structures_path, RHSₙi)))
      RBVars.nᵘ = size(RBVars.RHSₙ[i])[1]

    end

  end

  operators

end

function get_θᵃ(RBInfo::Info, RBVars::PoissonSteady, Param) ::Array

  if !RBInfo.probl_nl["A"]
    θᵃ = Param.α(Point(0.,0.))
  else
    A_μ_sparse = build_sparse_mat(FEMInfo, FEMSpace, Param, RBVars.sparse_el_A)
    θᵃ = M_DEIM_online(A_μ_sparse, RBVars.MDEIMᵢ_A, RBVars.MDEIM_idx_A)
  end

  return θᵃ

end

function get_θᶠʰ(RBInfo::Info, RBVars::PoissonSteady, Param) ::Tuple

  if RBInfo.build_Parametric_RHS
    error("Cannot fetch θᶠ, θʰ if the RHS is built online")
  end

  if !RBInfo.probl_nl["f"] && !RBInfo.probl_nl["h"]
    θᶠ, θʰ = Param.f(Point(0.,0.)), Param.h(Point(0.,0.))
  elseif !RBInfo.probl_nl["f"] && RBInfo.probl_nl["h"]
    H_μ = assemble_neumann_datum(FEMSpace, RBInfo, Param)
    θᶠ, θʰ = Param.f(Point(0.,0.)), M_DEIM_online(H_μ, RBVars.DEIMᵢ_mat_H, RBVars.DEIM_idx_H)
  elseif RBInfo.probl_nl["f"] && !RBInfo.probl_nl["h"]
    F_μ = assemble_forcing(FEMSpace, RBInfo, Param)
    θᶠ, θʰ = M_DEIM_online(F_μ, RBVars.DEIMᵢ_mat_F, RBVars.DEIM_idx_F), Param.h(Point(0.,0.))
  else RBInfo.probl_nl["f"] && RBInfo.probl_nl["h"]
    F_μ = assemble_forcing(FEMSpace, RBInfo, Param)
    H_μ = assemble_neumann_datum(FEMSpace, RBInfo, Param)
    θᶠ, θʰ = M_DEIM_online(F_μ, RBVars.DEIMᵢ_mat_F, RBVars.DEIM_idx_F), M_DEIM_online(H_μ, RBVars.DEIMᵢ_mat_H, RBVars.DEIM_idx_H)
  end

  return θᶠ, θʰ

end

function initialize_RB_system(RBVars::RBProblem)

  RBVars.LHSₙ = Matrix{Float64}[]
  RBVars.RHSₙ = Matrix{Float64}[]

end

function get_Q(RBInfo::Info, RBVars::PoissonSteady)

  if RBVars.Qᵃ == 0
    RBVars.Qᵃ = size(RBVars.Aₙ)[end]
  end

  if !RBInfo.build_Parametric_RHS
    if RBVars.Qᶠ == 0
      RBVars.Qᶠ = size(RBVars.Fₙ)[end]
    end
    if RBVars.Qʰ == 0
      RBVars.Qʰ = size(RBVars.Hₙ)[end]
    end
  end

end

function assemble_online_structure(θ::Array, Mat::Array)

  Mat_shape = size(Mat)
  Mat = reshape(Mat,:,Mat_shape[end])
  reshape(Mat*θ,Mat_shape[1:end-1])

end

function get_RB_system(RBInfo::Info, RBVars::PoissonSteady, Param)

  initialize_RB_system(RBVars)
  get_Q(RBInfo, RBVars)
  blocks = [1]
  operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

  @info "Preparing the RB system: fetching reduced LHS"
  θᵃ, θᶠ, θʰ = get_θ(RBInfo, RBVars, Param)

  if "LHS" ∈ operators
    Aₙ_μ = assemble_online_structure(θᵃ, RBVars.Aₙ)
    push!(RBVars.LHSₙ, Aₙ_μ)
  end

  if "RHS" ∈ operators
    if !RBInfo.build_Parametric_RHS
      @info "Preparing the RB system: fetching reduced RHS"
      Fₙ_μ = assemble_online_structure(θᶠ, RBVars.Fₙ)
      Hₙ_μ = assemble_online_structure(θʰ, RBVars.Hₙ)
      push!(RBVars.RHSₙ, reshape(Fₙ_μ+Hₙ_μ,:,1))
    else
      @info "Preparing the RB system: assembling reduced RHS exactly"
      Fₙ_μ, Hₙ_μ = build_Param_RHS(RBInfo, RBVars, Param, θᵃ)
      push!(RBVars.RHSₙ, reshape(Fₙ_μ+Hₙ_μ,:,1))
    end
  end

end

function solve_RB_system(RBInfo::Info, RBVars::PoissonSteady, Param)

  get_RB_system(RBInfo, RBVars, Param)

  @info "Solving RB problem via backslash"
  @info "Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))"
  RBVars.uₙ = zeros(RBVars.nₛᵘ)
  RBVars.uₙ = RBVars.LHSₙ[1] \ RBVars.RHSₙ[1]

end

function reconstruct_FEM_solution(RBVars::PoissonSteady)


  @info "Reconstructing FEM solution from the newly computed RB one"

  RBVars.ũ = RBVars.Φₛᵘ * RBVars.uₙ

end

function offline_phase(RBInfo::Info, RBVars::PoissonSteady)


  @info "Building $(RBInfo.RB_method) approximation with $(RBInfo.nₛ) snapshots and tolerances of $(RBInfo.ϵₛ) in space"

  if RBInfo.import_snapshots
    get_snapshot_matrix(RBInfo, RBVars)
    import_snapshots_success = true
  else
    import_snapshots_success = false
  end

  if RBInfo.import_offline_structures
    import_reduced_basis(RBInfo, RBVars)
    import_basis_success = true
  else
    import_basis_success = false
  end

  if !import_snapshots_success && !import_basis_success
    error("Impossible to assemble the reduced problem if
      neither the snapshots nor the bases can be loaded")
  end

  if import_snapshots_success && !import_basis_success
    @info "Failed to import the reduced basis, building it via POD"
    build_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.import_offline_structures
    operators = get_offline_structures(RBInfo, RBVars)
    if "A" ∈ operators || "F" ∈ operators || "H" ∈ operators
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

end

function online_phase(RBInfo::Info, RBVars::PoissonSteady, μ, Param_nbs)

  mean_H1_err = 0.0
  mean_pointwise_err = zeros(RBVars.Nₛᵘ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(RBInfo, RBVars)

  ũ_μ = zeros(RBVars.Nₛᵘ, length(Param_nbs))
  uₙ_μ = zeros(RBVars.nₛᵘ, length(Param_nbs))

  for nb in Param_nbs
    @info "Considering Parameter number: $nb"

    μ_nb = parse.(Float64, split(chop(μ[nb]; head=1, tail=1), ','))
    Param = get_ParamInfo(problem_ntuple, RBInfo, μ_nb)

    uₕ_test = Matrix(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"), DataFrame))[:, nb]

    online_time = @elapsed begin
      solve_RB_system(RBInfo, RBVars, Param)
    end
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    mean_online_time = online_time / length(Param_nbs)
    mean_reconstruction_time = reconstruction_time / length(Param_nbs)

    H1_err_nb = compute_errors(uₕ_test, RBVars, RBVars.Xᵘ₀)
    mean_H1_err += H1_err_nb / length(Param_nbs)
    mean_pointwise_err += abs.(uₕ_test - RBVars.ũ) / length(Param_nbs)

    ũ_μ[:, nb - Param_nbs[1] + 1] = RBVars.ũ
    uₙ_μ[:, nb - Param_nbs[1] + 1] = RBVars.uₙ

    @info "Online wall time: $online_time s (snapshot number $nb)"
    @info "Relative reconstruction H1-error: $H1_err_nb (snapshot number $nb)"

  end

  string_Param_nbs = "Params"
  for Param_nb in Param_nbs
    string_Param_nbs *= "_" * string(Param_nb)
  end
  path_μ = joinpath(RBInfo.paths.results_path, string_Param_nbs)

  if RBInfo.save_results

    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV([mean_H1_err], joinpath(path_μ, "H1_err.csv"))

    if !RBInfo.import_offline_structures
      times = Dict(RBVars.S.offline_time=>"off_time",
        mean_online_time=>"on_time", mean_reconstruction_time=>"rec_time")
    else
      times = Dict(mean_online_time=>"on_time",
        mean_reconstruction_time=>"rec_time")
    end
    CSV.write(joinpath(path_μ, "times.csv"),times)

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FEMSpace"=>FEMSpace, "mean_point_err_u"=>mean_pointwise_err)

  if RBInfo.postprocess
    post_process(RBInfo, pass_to_pp)
  end

end

function post_process(RBInfo::SteadyInfo, d::Dict)
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(MDEIM_Σ), MDEIM_Σ, "Decay singular values, MDEIM",
      ["σ"], "σ index", "σ value", RBInfo.paths.results_path; var="MDEIM_Σ")
  end
  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(joinpath(RBInfo.paths.ROM_structures_path, "DEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(DEIM_Σ), DEIM_Σ, "Decay singular values, DEIM",
      ["σ"], "σ index", "σ value", RBInfo.paths.results_path; var="DEIM_Σ")
  end

  FEMSpace = d["FEMSpace"]
  writevtk(FEMSpace.Ω, joinpath(d["path_μ"], "mean_point_err"),
  cellfields = ["err"=> FEFunction(FEMSpace.V, d["mean_point_err_u"])])

end
