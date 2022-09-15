include("RB.jl")
include("S-GRB_Poisson.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSteady{T}) where T

  println("Importing the snapshot matrix for field u,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵘ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
    DataFrame))[:, 1:RBInfo.nₛ]
  println("Dimension of snapshot matrix: $(size(Sᵘ))")
  RBVars.Sᵘ = Sᵘ
  RBVars.Nₛᵘ = size(Sᵘ)[1]
end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::PoissonSteady{T}) where T

  if check_norm_matrix(RBVars)
    println("Importing the norm matrix Xᵘ₀")
    Xᵘ₀ = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "Xᵘ₀.csv"))
    RBVars.Nₛᵘ = size(Xᵘ₀)[1]
    println("Dimension of norm matrix, field u: $(size(Xᵘ₀))")
    if RBInfo.use_norm_X
      RBVars.Xᵘ₀ = Xᵘ₀
    else
      RBVars.Xᵘ₀ = one(T)*sparse(I,RBVars.Nₛᵘ,RBVars.Nₛᵘ)
    end
  end

end

function check_norm_matrix(RBVars::PoissonSteady)
  isempty(RBVars.Xᵘ₀)
end

function PODs_space(
  RBInfo::Info,
  RBVars::PoissonSteady)

  println("Performing the spatial POD for field u, using a tolerance of $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars)
  RBVars.Φₛᵘ = POD(RBVars.Sᵘ, RBInfo.ϵₛ, RBVars.Xᵘ₀)
  (RBVars.Nₛᵘ, RBVars.nₛᵘ) = size(RBVars.Φₛᵘ)

end

function build_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSteady)

  RBVars.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
  end

  if RBInfo.save_offline_structures
    save_CSV(RBVars.Φₛᵘ, joinpath(RBInfo.ROM_structures_path,"Φₛᵘ.csv"))
  end

  return

end

function import_reduced_basis(
  RBInfo::Info,
  RBVars::PoissonSteady{T}) where T

  println("Importing the spatial reduced basis for field u")
  RBVars.Φₛᵘ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₛᵘ.csv"))
  (RBVars.Nₛᵘ, RBVars.nₛᵘ) = size(RBVars.Φₛᵘ)

end

function get_generalized_coordinates(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSteady,
  snaps=nothing)

  get_norm_matrix(RBInfo, RBVars)
  if isnothing(snaps) || maximum(snaps) > RBInfo.nₛ
    snaps = 1:RBInfo.nₛ
  end
  Φₛᵘ_normed = RBVars.Xᵘ₀*RBVars.Φₛᵘ
  RBVars.û = RBVars.Sᵘ[:,snaps]*Φₛᵘ_normed
  if RBInfo.save_offline_structures
    save_CSV(RBVars.û, joinpath(RBInfo.ROM_structures_path, "û.csv"))
  end

end

function set_operators(
  RBInfo::Info,
  ::PoissonSteady)

  operators = ["A"]
  if !RBInfo.build_parametric_RHS
    append!(operators, ["F","H","L"])
  end
  operators

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSteady,
  var::String)

  println("The matrix $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
  if var == "A"
    if isempty(RBVars.MDEIM_mat_A)
      (RBVars.MDEIM_mat_A, RBVars.MDEIM_idx_A, RBVars.MDEIMᵢ_A,
      RBVars.row_idx_A,RBVars.sparse_el_A) = MDEIM_offline(RBInfo, RBVars, "A")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_A, RBVars.row_idx_A, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSteady,
  var::String)

  println("The vector $var is non-affine:
    running the DEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "F"
    if isempty(RBVars.DEIM_mat_F)
      RBVars.DEIM_mat_F, RBVars.DEIM_idx_F, RBVars.DEIMᵢ_F, RBVars.sparse_el_F =
        DEIM_offline(RBInfo,"F")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_F,"F")
  elseif var == "H"
    if isempty(RBVars.DEIM_mat_H)
      RBVars.DEIM_mat_H, RBVars.DEIM_idx_H, RBVars.DEIMᵢ_H, RBVars.sparse_el_H =
        DEIM_offline(RBInfo,"H")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_H,"H")
  elseif var == "L"
    if isempty(RBVars.DEIM_mat_L)
      RBVars.DEIM_mat_L, RBVars.DEIM_idx_L, RBVars.DEIMᵢ_L, RBVars.sparse_el_L =
        DEIM_offline(RBInfo,"L")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_L,"L")
  else
    error("Unrecognized variable on which to perform DEIM")
  end

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::PoissonSteady)

  affine_vars = (reshape(RBVars.Aₙ, :, RBVars.Qᵃ)::Matrix{T},
                RBVars.Fₙ, RBVars.Hₙ, RBVars.Lₙ)
  affine_names = ("Aₙ", "Fₙ", "Hₙ", "Lₙ")
  save_structures_in_list(affine_vars, affine_names, RBInfo.ROM_structures_path)

  M_DEIM_vars = (RBVars.MDEIM_mat_A, RBVars.MDEIMᵢ_A, RBVars.MDEIM_idx_A,
    RBVars.row_idx_A, RBVars.sparse_el_A, RBVars.DEIM_mat_F, RBVars.DEIMᵢ_F,
    RBVars.DEIM_idx_F, RBVars.sparse_el_F, RBVars.DEIM_mat_H, RBVars.DEIMᵢ_H,
    RBVars.DEIM_idx_H, RBVars.sparse_el_H, RBVars.DEIM_mat_L, RBVars.DEIMᵢ_L,
    RBVars.DEIM_idx_L, RBVars.sparse_el_L)
  M_DEIM_names = (
    "MDEIM_mat_A","MDEIMᵢ_A","MDEIM_idx_A","row_idx_A","sparse_el_A",
    "DEIM_mat_F","DEIMᵢ_F","DEIM_idx_F","sparse_el_F",
    "DEIM_mat_H","DEIMᵢ_H","DEIM_idx_H","sparse_el_H",
    "DEIM_mat_L","DEIMᵢ_L","DEIM_idx_L","sparse_el_L")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path)

end

function get_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSteady)

  operators = String[]

  append!(operators, get_A(RBInfo, RBVars))

  if RBInfo.build_parametric_RHS
    append!(operators, get_F(RBInfo, RBVars))
    append!(operators, get_H(RBInfo, RBVars))
    append!(operators, get_L(RBInfo, RBVars))
  end

  operators

end

function assemble_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSteady,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.offline_time += @elapsed begin
    for var ∈ intersect(operators, RBInfo.probl_nl)
      if var == "A"
        assemble_MDEIM_matrices(RBInfo, RBVars, var)
      else
        assemble_DEIM_vectors(RBInfo, RBVars, var)
      end
    end

    for var ∈ setdiff(operators, RBInfo.probl_nl)
      if var == "A"
        assemble_affine_matrices(RBInfo, RBVars, var)
      else
        assemble_affine_vectors(RBInfo, RBVars, var)
      end
    end
  end

  save_assembled_structures(RBInfo, RBVars)

end

function get_system_blocks(
  RBInfo::Info,
  RBVars::PoissonSteady{T},
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int}) where T

  if !RBInfo.import_offline_structures
    return ["LHS", "RHS"]
  end

  operators = String[]

  for i = LHS_blocks
    LHSₙi = "LHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(RBInfo.ROM_structures_path, LHSₙi * ".csv"))
      append!(operators, ["LHS"])
      break
    end
  end
  for i = RHS_blocks
    RHSₙi = "RHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(RBInfo.ROM_structures_path, RHSₙi * ".csv"))
      append!(operators, ["RHS"])
      break
    end
  end
  if "LHS" ∉ operators
    for i = LHS_blocks
      LHSₙi = "LHSₙ" * string(i) * ".csv"
      println("Importing block number $i of the reduced affine LHS")
      push!(RBVars.LHSₙ,
        load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, LHSₙi)))
    end
  end
  if "RHS" ∉ operators
    for i = RHS_blocks
      RHSₙi = "RHSₙ" * string(i) * ".csv"
      println("Importing block number $i of the reduced affine RHS")
      push!(RBVars.RHSₙ,
        load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, RHSₙi)))
    end
  end

  operators

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::PoissonSteady{T},
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String}) where T

  if "A" ∉ RBInfo.probl_nl && "LHS" ∈ operators
    for i = LHS_blocks
      LHSₙi = "LHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.LHSₙ[i],joinpath(RBInfo.ROM_structures_path, LHSₙi))
    end
  end
  if "F" ∉ RBInfo.probl_nl && "H" ∉ RBInfo.probl_nl && "RHS" ∈ operators
    for i = RHS_blocks
      RHSₙi = "RHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.RHSₙ[i],joinpath(RBInfo.ROM_structures_path, RHSₙi))
    end
  end
end

function get_θ_matrix(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::PoissonSteady,
  Param::SteadyParametricInfo,
  var::String) where T

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param.α, RBVars.MDEIMᵢ_A,
      RBVars.MDEIM_idx_A, RBVars.sparse_el_A, "A")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_θ_vector(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::PoissonSteady,
  Param::SteadyParametricInfo,
  var::String) where T

  if var == "F"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.f, RBVars.DEIMᵢ_F,
      RBVars.DEIM_idx_F, RBVars.sparse_el_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.h, RBVars.DEIMᵢ_H,
      RBVars.DEIM_idx_H, RBVars.sparse_el_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.g, RBVars.DEIMᵢ_L,
      RBVars.DEIM_idx_L, RBVars.sparse_el_L, "L")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_θ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSGRB{T},
  Param::SteadyParametricInfo) where T

  θᵃ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "A")

  if !RBInfo.build_parametric_RHS
    θᶠ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "F")
    θʰ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "H")
    θˡ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "L")
  else
    θᶠ, θʰ, θˡ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  end

  return θᵃ, θᶠ, θʰ, θˡ

end

function get_RB_LHS_blocks(
  RBVars::PoissonSteady{T},
  θᵃ::Matrix) where T

  println("Assembling reduced LHS")

  block₁ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ)
  for q = 1:RBVars.Qᵃ
    block₁ += RBVars.Aₙ[:,:,q] * θᵃ[q]
  end

  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBVars::PoissonSteady{T},
  θᶠ::Matrix,
  θʰ::Matrix,
  θˡ::Matrix) where T

  println("Assembling reduced RHS")

  block₁ = RBVars.Fₙ * θᶠ + RBVars.Hₙ * θʰ - RBVars.Lₙ * θˡ

  push!(RBVars.RHSₙ, block₁)::Vector{Matrix{T}}

end

function initialize_RB_system(RBVars::PoissonSteady{T}) where T
  RBVars.LHSₙ = Matrix{T}[]
  RBVars.RHSₙ = Matrix{T}[]
end

function initialize_online_time(RBVars::PoissonSteady)
  RBVars.online_time = 0.0
end

function solve_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSteady,
  Param::SteadyParametricInfo)

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")
  RBVars.online_time += @elapsed begin
    RBVars.uₙ = RBVars.LHSₙ[1] \ RBVars.RHSₙ[1]
  end

end

function reconstruct_FEM_solution(RBVars::PoissonSteady)
  println("Reconstructing FEM solution from the newly computed RB one")
  RBVars.ũ = RBVars.Φₛᵘ * RBVars.uₙ
end

function offline_phase(
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSteady)

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
  RBVars::PoissonSteady{T},
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

    Param = get_ParamInfo(RBInfo, μ[nb])

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
  path_μ = joinpath(RBInfo.results_path, string_Param_nbs)

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
      "on_time"=>mean_online_time,"rec_time"=>mean_reconstruction_time)

    CSV.write(joinpath(path_μ, "times.csv"),times)

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FEMSpace"=>FEMSpace,
    "mean_point_err_u"=>Float.(mean_pointwise_err))

  if RBInfo.post_process
    post_process(RBInfo, pass_to_pp)
  end

end
