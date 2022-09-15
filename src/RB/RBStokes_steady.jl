include("RBPoisson_steady.jl")
include("S-GRB_Stokes.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady{T}) where T

  get_snapshot_matrix(RBInfo, RBVars.Poisson)

  println("Importing the snapshot matrix for field p,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵖ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
    DataFrame))[:, 1:RBInfo.nₛ]
  println("Dimension of pressure snapshot matrix: $(size(Sᵖ))")
  RBVars.Sᵖ = Sᵖ
  RBVars.Nₛᵖ = size(Sᵖ)[1]

end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::StokesSteady{T}) where T

  get_norm_matrix(RBInfo, RBVars.Poisson)

  if check_norm_matrix(RBVars)

    println("Importing the norm matrix Xᵖ₀")

    Xᵖ₀ = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "Xᵖ₀.csv"))
    RBVars.Nₛᵖ = size(Xᵖ₀)[1]
    println("Dimension of L² norm matrix, field p: $(size(Xᵖ₀))")

    if RBInfo.use_norm_X
      RBVars.Xᵖ₀ = Xᵖ₀
    else
      RBVars.Xᵖ₀ = one(T)*sparse(I,RBVars.Nₛᵖ,RBVars.Nₛᵖ)
    end

  end

end

function check_norm_matrix(RBVars::StokesSteady)
  isempty(RBVars.Xᵘ₀) || isempty(RBVars.Xᵖ₀)
end

function PODs_space(
  RBInfo::Info,
  RBVars::StokesSteady)

  PODs_space(RBInfo,RBVars.Poisson)

  println("Performing the spatial POD for field p, using a tolerance of $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars)
  RBVars.Φₛᵖ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.Xᵖ₀)
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function primal_supremizers(
  RBInfo::Info,
  RBVars::StokesSteady{T}) where T

  println("Computing primal supremizers")

  constraint_mat = load_CSV(sparse([],[],T[]),
    joinpath(get_FEM_structures_path(RBInfo), "B.csv"))'

  supr_primal = Matrix{T}(RBVars.Xᵘ₀) \ (Matrix{T}(constraint_mat) * RBVars.Φₛᵖ)

  min_norm = 1e16
  for i = 1:size(supr_primal)[2]

    println("Normalizing primal supremizer $i")

    for j in 1:RBVars.nₛᵘ
      supr_primal[:, i] -= mydot(supr_primal[:, i], RBVars.Φₛᵘ[:,j], RBVars.Xᵘ₀) /
      mynorm(RBVars.Φₛᵘ[:,j], RBVars.Xᵘ₀) * RBVars.Φₛᵘ[:,j]
    end
    for j in 1:i
      supr_primal[:, i] -= mydot(supr_primal[:, i], supr_primal[:, j], RBVars.Xᵘ₀) /
      mynorm(supr_primal[:, j], RBVars.Xᵘ₀) * supr_primal[:, j]
    end

    supr_norm = mynorm(supr_primal[:, i], RBVars.Xᵘ₀)
    min_norm = min(supr_norm, min_norm)
    println("Norm supremizers: $supr_norm")
    supr_primal[:, i] /= supr_norm

  end

  println("Primal supremizers enrichment ended with norm: $min_norm")

  supr_primal

end

function supr_enrichment_space(
  RBInfo::Info,
  RBVars::StokesSteady)

  supr_primal = primal_supremizers(RBInfo, RBVars)
  RBVars.Φₛᵘ = hcat(RBVars.Φₛᵘ, supr_primal)
  RBVars.nₛᵘ = size(RBVars.Φₛᵘ)[2]

end

function build_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady)

  RBVars.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
    supr_enrichment_space(RBInfo, RBVars)
  end

  if RBInfo.save_offline_structures
    save_CSV(RBVars.Φₛᵘ, joinpath(RBInfo.ROM_structures_path,"Φₛᵘ.csv"))
    save_CSV(RBVars.Φₛᵖ, joinpath(RBInfo.ROM_structures_path,"Φₛᵖ.csv"))
  end

  return

end

function import_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady{T}) where T

  import_reduced_basis(RBInfo, RBVars.Poisson)

  println("Importing the spatial reduced basis for field p")
  RBVars.Φₛᵖ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₛᵖ.csv"))
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function get_generalized_coordinates(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  snaps=nothing)

  get_norm_matrix(RBInfo, RBVars)
  if isnothing(snaps) || maximum(snaps) > RBInfo.nₛ
    snaps = 1:RBInfo.nₛ
  end

  get_generalized_coordinates(RBInfo, RBVars.Poisson, snaps)

  Φₛᵖ_normed = RBVars.Xᵖ₀*RBVars.Φₛᵖ
  RBVars.p̂ = RBVars.Sᵖ[:,snaps]*Φₛᵖ_normed
  if RBInfo.save_offline_structures
    save_CSV(RBVars.p̂, joinpath(RBInfo.ROM_structures_path, "p̂.csv"))
  end

end

function set_operators(
  RBInfo::Info,
  RBVars::StokesSteady)

  append!(["B", "Lc"], set_operators(RBInfo, RBVars.Poisson))

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  var::String)

  println("The matrix $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
  if var == "A"
    if isempty(RBVars.MDEIM_mat_A)
      (RBVars.MDEIM_mat_A, RBVars.MDEIM_idx_A, RBVars.MDEIMᵢ_A,
      RBVars.row_idx_A,RBVars.sparse_el_A) = MDEIM_offline(RBInfo, RBVars, "A")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_A, RBVars.row_idx_A, var)
  elseif var == "B"
    if isempty(RBVars.MDEIM_mat_B)
      (RBVars.MDEIM_mat_B, RBVars.MDEIM_idx_B, RBVars.MDEIMᵢ_B,
      RBVars.row_idx_B,RBVars.sparse_el_B) = MDEIM_offline(RBInfo, RBVars, "B")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_B, RBVars.row_idx_B, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
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
  elseif var == "Lc"
    if isempty(RBVars.DEIM_mat_Lc)
      RBVars.DEIM_mat_Lc, RBVars.DEIM_idx_Lc, RBVars.DEIMᵢ_Lc, RBVars.sparse_el_Lc =
        DEIM_offline(RBInfo,"Lc")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_Lc,"Lc")
  else
    error("Unrecognized variable on which to perform DEIM")
  end

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::PoissonSteady)

  affine_vars = (reshape(RBVars.Bₙ, :, RBVars.Qᵇ)::Matrix{T}, RBVars.Lcₙ)
  affine_names = ("Bₙ", "Lcₙ")
  save_structures_in_list(affine_vars, affine_names, RBInfo.ROM_structures_path)

  M_DEIM_vars = (
    RBVars.MDEIM_mat_B, RBVars.MDEIMᵢ_B, RBVars.MDEIM_idx_B, RBVars.row_idx_B,
    RBVars.sparse_el_B, RBVars.DEIM_mat_Lc, RBVars.DEIMᵢ_Lc, RBVars.DEIM_idx_Lc,)
    RBVars.sparse_el_Lc
  M_DEIM_names = (
    "MDEIM_mat_B","MDEIMᵢ_B","MDEIM_idx_B","row_idx_B","sparse_el_B",
    "DEIM_mat_Lc","DEIMᵢ_Lc","DEIM_idx_Lc","sparse_el_Lc")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path)

  save_assembled_structures(RBInfo, RBVars.Poisson)

end

function get_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady)

  operators = String[]

  append!(operators, get_A(RBInfo, RBVars))
  append!(operators, get_B(RBInfo, RBVars))

  if RBInfo.build_parametric_RHS
    append!(operators, get_F(RBInfo, RBVars))
    append!(operators, get_H(RBInfo, RBVars))
    append!(operators, get_L(RBInfo, RBVars))
    append!(operators, get_Lc(RBInfo, RBVars))
  end

  operators

end

function get_system_blocks(
  RBInfo::Info,
  RBVars::StokesSteady,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int})

  get_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::StokesSteady,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  save_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks, operators)

end

function get_θ_matrix(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  Param::SteadyParametricInfo,
  var::String)

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param.α, RBVars.MDEIMᵢ_A,
      RBVars.MDEIM_idx_A, RBVars.sparse_el_A, "A")::Matrix{T}
  elseif var == "B"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param.b, RBVars.MDEIMᵢ_B,
      RBVars.MDEIM_idx_B, RBVars.sparse_el_B, "B")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_θ_vector(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  Param::SteadyParametricInfo,
  var::String)

  if var == "F"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.f, RBVars.DEIMᵢ_F,
      RBVars.DEIM_idx_F, RBVars.sparse_el_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.h, RBVars.DEIMᵢ_H,
      RBVars.DEIM_idx_H, RBVars.sparse_el_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.g, RBVars.DEIMᵢ_L,
      RBVars.DEIM_idx_L, RBVars.sparse_el_L, "L")::Matrix{T}
  elseif var == "Lc"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.g, RBVars.DEIMᵢ_Lc,
      RBVars.DEIM_idx_Lc, RBVars.sparse_el_Lc, "Lc")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_θ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSGRB{T},
  Param::SteadyParametricInfo) where T

  θᵃ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "A")
  θᵇ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "B")

  if !RBInfo.build_parametric_RHS
    θᶠ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "F")
    θʰ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "H")
    θˡ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "L")
    θˡᶜ = get_θ_vector(FEMSpace, RBInfo, RBVars, Param, "Lc")
  else
    θᶠ, θʰ, θˡ, θˡᶜ = (Matrix{T}(undef,0,0), Matrix{T}(undef,0,0),
      Matrix{T}(undef,0,0), Matrix{T}(undef,0,0))
  end

  return θᵃ, θᵇ, θᶠ, θʰ, θˡ, θˡᶜ

end

function initialize_RB_system(RBVars::StokesSteady)
  initialize_RB_system(RBVars.Poisson)
end

function initialize_online_time(RBVars::StokesSteady)
  initialize_online_time(RBVars.Poisson)
end

function get_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  Param::SteadyParametricInfo)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    LHS_blocks = [1, 2, 3]
    RHS_blocks = [1, 2]
    operators = get_system_blocks(RBInfo, RBVars, LHS_blocks, RHS_blocks)

    θᵃ, θᵇ, θᶠ, θʰ, θˡ, θˡᶜ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBVars, θᵃ, θᵇ)
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        get_RB_RHS_blocks(RBVars, θᶠ, θʰ, θˡ, θˡᶜ)
      else
        build_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,LHS_blocks,RHS_blocks,operators)

end

function solve_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::StokesSteady,
  Param::SteadyParametricInfo) where T

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)
  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")
  RBVars.online_time += @elapsed begin
    xₙ = (vcat(hcat(RBVars.LHSₙ[1], RBVars.LHSₙ[2]),
      hcat(RBVars.LHSₙ[3], zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ))) \
      vcat(RBVars.RHSₙ[1], RBVars.RHSₙ[2]))
  end

  RBVars.uₙ = xₙ[1:RBVars.nₛᵘ,:]
  RBVars.pₙ = xₙ[RBVars.nₛᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::StokesSteady)
  println("Reconstructing FEM solution from the newly computed RB one")
  reconstruct_FEM_solution(RBVars.Poisson)
  RBVars.p̃ = RBVars.Φₛᵖ * RBVars.pₙ
end

function offline_phase(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady)

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
  RBVars::StokesSteady{T},
  param_nbs) where T

  μ = load_CSV(Array{T}[],
    joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)

  mean_H1_err = 0.0
  mean_L2_err = 0.0
  mean_pointwise_err_u = zeros(T, RBVars.Nₛᵘ)
  mean_pointwise_err_p = zeros(T, RBVars.Nₛᵖ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(RBInfo, RBVars)

  ũ_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs))
  uₙ_μ = zeros(T, RBVars.nₛᵘ, length(param_nbs))
  p̃_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs))
  pₙ_μ = zeros(T, RBVars.nₛᵘ, length(param_nbs))

  for nb in param_nbs
    println("Considering parameter number: $nb")

    Param = get_ParamInfo(RBInfo, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:, nb]
    pₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
      DataFrame))[:, nb]

    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    mean_online_time = RBVars.online_time / length(param_nbs)
    mean_reconstruction_time = reconstruction_time / length(param_nbs)

    H1_err_nb = compute_errors(uₕ_test, RBVars, RBVars.Xᵘ₀)
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test - RBVars.ũ) / length(param_nbs)

    L2_err_nb = compute_errors(pₕ_test, RBVars, RBVars.Xᵖ₀)
    mean_L2_err += L2_err_nb / length(param_nbs)
    mean_pointwise_err_p += abs.(pₕ_test - RBVars.p̃) / length(param_nbs)

    ũ_μ[:, nb - param_nbs[1] + 1] = RBVars.ũ
    uₙ_μ[:, nb - param_nbs[1] + 1] = RBVars.uₙ
    p̃_μ[:, nb - param_nbs[1] + 1] = RBVars.p̃
    pₙ_μ[:, nb - param_nbs[1] + 1] = RBVars.pₙ

    println("Online wall time: $(RBVars.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-error: $H1_err_nb (snapshot number $nb)")
    println("Relative reconstruction L2-error: $L2_err_nb (snapshot number $nb)")

  end

  string_param_nbs = "params"
  for Param_nb in param_nbs
    string_param_nbs *= "_" * string(Param_nb)
  end
  path_μ = joinpath(RBInfo.results_path, string_param_nbs)

  if RBInfo.save_results

    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err_u, joinpath(path_μ, "mean_point_err_u.csv"))
    save_CSV([mean_H1_err], joinpath(path_μ, "H1_err.csv"))
    save_CSV(p̃_μ, joinpath(path_μ, "p̃.csv"))
    save_CSV(pₙ_μ, joinpath(path_μ, "pₙ.csv"))
    save_CSV(mean_pointwise_err_p, joinpath(path_μ, "mean_point_err_p.csv"))
    save_CSV([mean_L2_err], joinpath(path_μ, "L2_err.csv"))

    if RBInfo.import_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time,"rec_time"=>mean_reconstruction_time)

    CSV.write(joinpath(path_μ, "times.csv"),times)

  end

  pass_to_pp = Dict("path_μ"=>path_μ, "FEMSpace"=>FEMSpace,
    "mean_point_err_u"=>Float.(mean_pointwise_err_u),
    "mean_point_err_p"=>Float.(mean_pointwise_err_p))

  if RBInfo.post_process
    post_process(RBInfo, pass_to_pp)
  end

end
