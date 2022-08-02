include("RB.jl")
include("S-GRB_ADR.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoSteady,
  RBVars::ADRSteady)

  get_snapshot_matrix(RBInfo, RBVars.Poisson)

end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::ADRSteady)

  get_norm_matrix(RBInfo, RBVars.Poisson)

end

function build_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::ADRSteady)

  build_reduced_basis(RBInfo, RBVars.Poisson)

end

function import_reduced_basis(
  RBInfo::Info,
  RBVars::ADRSteady)

  import_reduced_basis(RBInfo, RBVars.Poisson)

end

function set_operators(
  RBInfo::Info,
  RBVars::ADRSteady)

  vcat(["B","R"], set_operators(RBInfo, RBVars.Poisson))

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoSteady,
  RBVars::ADRSteady,
  var::String)

  if var == "B"
    println("The matrix B is non-affine:
      running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
    if isempty(RBVars.MDEIM_mat_B)
      (RBVars.MDEIM_mat_B, RBVars.MDEIM_idx_B, RBVars.MDEIMᵢ_B,
      RBVars.row_idx_B,RBVars.sparse_el_B) = MDEIM_offline(RBInfo, RBVars, "B")
    end
    assemble_reduced_mat_MDEIM(RBVars,RBVars.MDEIM_mat_B,RBVars.row_idx_B)
  elseif var == "D"
    println("The matrix D is non-affine:
      running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
    if isempty(RBVars.MDEIM_mat_D)
      (RBVars.MDEIM_mat_D, RBVars.MDEIM_idx_D, RBVars.MDEIMᵢ_D,
      RBVars.row_idx_D,RBVars.sparse_el_D) = MDEIM_offline(RBInfo, RBVars, "D")
    end
    assemble_reduced_mat_MDEIM(RBVars,RBVars.MDEIM_mat_D,RBVars.row_idx_D)
  else
    assemble_MDEIM_matrices(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoSteady,
  RBVars::ADRSteady,
  var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.Poisson, var)

end

function save_M_DEIM_structures(
  RBInfo::Info,
  RBVars::ADRSteady)

  list_M_DEIM = (RBVars.MDEIM_mat_B, RBVars.MDEIMᵢ_B, RBVars.MDEIM_idx_B,
    RBVars.row_idx_B, RBVars.sparse_el_B, RBVars.MDEIM_mat_D, RBVars.MDEIMᵢ_D,
    RBVars.MDEIM_idx_D, RBVars.row_idx_D, RBVars.sparse_el_D)
  list_names = ("MDEIM_mat_B","MDEIMᵢ_B","MDEIM_idx_B","row_idx_B","sparse_el_B",
  "MDEIM_mat_D","MDEIMᵢ_D","MDEIM_idx_D","row_idx_D","sparse_el_D")

  save_structures_in_list(list_M_DEIM, list_names,
    RBInfo.Paths.ROM_structures_path)

end

function get_M_DEIM_structures(
  RBInfo::Info,
  RBVars::ADRSteady{T}) where T

  operators = String[]
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars.Poisson))

  if RBInfo.probl_nl["B"]

    if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "MDEIMᵢ_B.csv"))
      println("Importing MDEIM offline structures, B")
      RBVars.MDEIMᵢ_B = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "MDEIMᵢ_B.csv"))
      RBVars.MDEIM_idx_B = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "MDEIM_idx_B.csv"))
      RBVars.row_idx_B = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "row_idx_B.csv"))
      RBVars.sparse_el_B = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "sparse_el_B.csv"))
    else
      println("Failed to import MDEIM offline structures,
        B: must build them")
      append!(operators, ["B"])
    end

  end

  if RBInfo.probl_nl["D"]

    if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "MDEIMᵢ_D.csv"))
      println("Importing MDEIM offline structures, D")
      RBVars.MDEIMᵢ_D = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "MDEIMᵢ_D.csv"))
      RBVars.MDEIM_idx_D = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "MDEIM_idx_D.csv"))
      RBVars.row_idx_D = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "row_idx_D.csv"))
      RBVars.sparse_el_D = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "sparse_el_D.csv"))
    else
      println("Failed to import MDEIM offline structures,
        D: must build them")
      append!(operators, ["D"])
    end

  end

  operators

end

function get_Fₙ(
  RBInfo::Info,
  RBVars::ADRSteady{T}) where T

  get_Fₙ(RBInfo, RBVars.Poisson)

end

function get_Hₙ(
  RBInfo::Info,
  RBVars::ADRSteady{T}) where T

  get_Hₙ(RBInfo, RBVars.Poisson)

end

function get_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::ADRSteady)

  operators = String[]

  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_system_blocks(
  RBInfo::Info,
  RBVars::ADRSteady{T},
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int}) where T

  get_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::ADRSteady{T},
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String}) where T

  if (!RBInfo.probl_nl["A"] && !RBInfo.probl_nl["B"]
      && !RBInfo.probl_nl["D"] && "LHS" ∈ operators)
    for i = LHS_blocks
      LHSₙi = "LHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.LHSₙ[i],joinpath(RBInfo.Paths.ROM_structures_path, LHSₙi))
    end
  end
  if !RBInfo.probl_nl["f"] && !RBInfo.probl_nl["h"] && "RHS" ∈ operators
    for i = RHS_blocks
      RHSₙi = "RHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.RHSₙ[i],joinpath(RBInfo.Paths.ROM_structures_path, RHSₙi))
    end
  end

end

function get_θᵃ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::ADRSteady,
  Param::ParametricInfoSteady) where T

  get_θᵃ(FEMSpace, RBInfo, RBVars.Poisson, Param)

end

function get_θᵇ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::ADRSteady,
  Param::ParametricInfoSteady) where T

  if !RBInfo.probl_nl["B"]
    θᵇ = reshape([T.(Param.b(Point(0.,0.)))],1,1)
  else
    B_μ_sparse = T.(build_sparse_mat(FEMSpace, FEMInfo, Param, RBVars.sparse_el_B))
    θᵇ = M_DEIM_online(B_μ_sparse, RBVars.MDEIMᵢ_B, RBVars.MDEIM_idx_B)
  end

  θᵇ::Matrix{T}

end

function get_θᵈ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::ADRSteady,
  Param::ParametricInfoSteady) where T

  if !RBInfo.probl_nl["D"]
    θᵈ = reshape([T.(Param.σ(Point(0.,0.)))],1,1)
  else
    D_μ_sparse = T.(build_sparse_mat(FEMSpace, FEMInfo, Param, RBVars.sparse_el_D))
    θᵈ = M_DEIM_online(D_μ_sparse, RBVars.MDEIMᵢ_D, RBVars.MDEIM_idx_D)
  end

  θᵈ::Matrix{T}

end

function get_θᶠʰ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::ADRSteady,
  Param::ParametricInfoSteady) where T

  get_θᶠʰ(FEMSpace, RBInfo, RBVars.Poisson, Param)

end

function get_RB_LHS_blocks(
  RBVars::ADRSteady{T},
  θᵃ::Matrix,
  θᵇ::Matrix,
  θᵈ::Matrix) where T

  println("Assembling reduced LHS")

  block₁ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ)
  for q = 1:RBVars.Qᵃ
    block₁ += RBVars.Aₙ[:,:,q] * θᵃ[q]
  end
  for q = 1:RBVars.Qᵇ
    block₁ += RBVars.Bₙ[:,:,q] * θᵇ[q]
  end
  for q = 1:RBVars.Qᵈ
    block₁ += RBVars.Dₙ[:,:,q] * θᵈ[q]
  end

  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBVars::ADRSteady{T},
  θᶠ::Array,
  θʰ::Array) where T

  get_RB_RHS_blocks(RBVars.Poisson, θᶠ, θʰ)

end

function initialize_RB_system(RBVars::ADRSteady{T}) where T
  initialize_RB_system(RBVars.Poisson)
end

function initialize_online_time(RBVars::ADRSteady)
  initialize_online_time(RBVars.Poisson)
end

function solve_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::ADRSteady,
  Param::ParametricInfoSteady)

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)
  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")
  RBVars.online_time += @elapsed begin
    RBVars.uₙ = RBVars.LHSₙ[1] \ RBVars.RHSₙ[1]
  end

end

function reconstruct_FEM_solution(RBVars::ADRSteady)
  reconstruct_FEM_solution(RBVars.Poisson)
end

function offline_phase(
  RBInfo::ROMInfoSteady,
  RBVars::ADRSteady)

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
    println("Failed to import the reduced basis, building it via POD")
    build_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.import_offline_structures
    operators = get_offline_structures(RBInfo, RBVars)
    if !isempty(operators)
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

end

function online_phase(
  RBInfo::ROMInfoSteady,
  RBVars::ADRSteady{T},
  Param_nbs) where T

  μ = load_CSV(Array{T}[],
    joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)

  mean_H1_err = 0.0
  mean_pointwise_err = zeros(T, RBVars.Nₛᵘ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(RBInfo, RBVars)

  ũ_μ = zeros(T, RBVars.Nₛᵘ, length(Param_nbs))
  uₙ_μ = zeros(T, RBVars.nₛᵘ, length(Param_nbs))

  for nb in Param_nbs
    println("Considering parameter number: $nb")

    Param = get_ParamInfo(RBInfo, FEMSpace, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"), DataFrame))[:, nb]

    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    mean_online_time = RBVars.online_time / length(Param_nbs)
    mean_reconstruction_time = reconstruction_time / length(Param_nbs)

    H1_err_nb = compute_errors(uₕ_test, RBVars, RBVars.Xᵘ₀)
    mean_H1_err += H1_err_nb / length(Param_nbs)
    mean_pointwise_err += abs.(uₕ_test - RBVars.ũ) / length(Param_nbs)

    ũ_μ[:, nb - Param_nbs[1] + 1] = RBVars.ũ
    uₙ_μ[:, nb - Param_nbs[1] + 1] = RBVars.uₙ

    println("Online wall time: $(RBVars.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-error: $H1_err_nb (snapshot number $nb)")

  end

  string_Param_nbs = "params"
  for Param_nb in Param_nbs
    string_Param_nbs *= "_" * string(Param_nb)
  end
  path_μ = joinpath(RBInfo.Paths.results_path, string_Param_nbs)

  if RBInfo.save_results

    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV([mean_H1_err], joinpath(path_μ, "H1_err.csv"))

    if RBInfo.import_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time+adapt_time,"rec_time"=>mean_reconstruction_time)

    CSV.write(joinpath(path_μ, "times.csv"),times)

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FEMSpace"=>FEMSpace,
    "mean_point_err_u"=>Float.(mean_pointwise_err))

  if RBInfo.post_process
    post_process(RBInfo, pass_to_pp)
  end

end
