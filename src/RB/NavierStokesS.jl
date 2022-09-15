include("StokesS.jl")
include("NavierStokesS_support.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady{T}) where T

  get_snapshot_matrix(RBInfo, RBVars.Stokes)

  println("Importing the snapshot matrix for field u on quadrature points,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵘ_quad = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ_quadp.csv"),
    DataFrame))[:, 1:RBInfo.nₛ]
  println("Dimension of velocity snapshot matrix on quadrature points: $(size(Sᵘ_quad))")
  RBVars.Nₛᵘ_quad = size(Sᵘ_quad)[1]

end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::NavierStokesSteady{T}) where T

  get_norm_matrix(RBInfo, RBVars.Stokes)

end

function PODs_space(
  RBInfo::Info,
  RBVars::NavierStokesSteady)

  PODs_space(RBInfo,RBVars.Stokes)

  println("Performing the spatial POD for field u on quadrature points")
  RBVars.Φₛᵘ_quad = POD(RBVars.Sᵘ_quad, RBInfo.ϵₛ, RBVars.Xᵘ₀)
  (RBVars.Nₛᵘ_quad, RBVars.nₛᵘ_quad) = size(RBVars.Φₛᵘ_quad)

end

function primal_supremizers(
  RBInfo::Info,
  RBVars::NavierStokesSteady{T}) where T

  println("Computing primal supremizers")

  constraint_mat = load_CSV(sparse([],[],T[]),
    joinpath(get_FEM_structures_path(RBInfo), "B.csv"))'

  supr_primal = Matrix{T}(RBVars.Xᵘ) \ (Matrix{T}(constraint_mat) * RBVars.Φₛᵖ)
  supr_primal_quad = supr_primal

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

  min_norm = 1e16
  for i = 1:size(supr_primal_quad)[2]

    println("Normalizing primal supremizer $i, quadrature points")

    for j in 1:RBVars.nₛᵘ_quad
      supr_primal_quad[:, i] -= mydot(supr_primal_quad[:, i], RBVars.Φₛᵘ_quad[:,j], RBVars.Xᵘ₀) /
      mynorm(RBVars.Φₛᵘ_quad[:,j], RBVars.Xᵘ₀) * RBVars.Φₛᵘ_quad[:,j]
    end
    for j in 1:i
      supr_primal_quad[:, i] -= mydot(supr_primal_quad[:, i], supr_primal_quad[:, j], RBVars.Xᵘ₀) /
      mynorm(supr_primal_quad[:, j], RBVars.Xᵘ₀) * supr_primal_quad[:, j]
    end

    supr_norm = mynorm(supr_primal_quad[:, i], RBVars.Xᵘ₀)
    min_norm = min(supr_norm, min_norm)
    println("Norm supremizers: $supr_norm")
    supr_primal_quad[:, i] /= supr_norm

  end

  println("Primal supremizers on quadrature points enrichment ended with norm: $min_norm")

  supr_primal, supr_primal_quad

end

function supr_enrichment_space(
  RBInfo::Info,
  RBVars::NavierStokesSteady)

  supr_primal, supr_primal_quad = primal_supremizers(RBInfo, RBVars)

  RBVars.Φₛᵘ = hcat(RBVars.Φₛᵘ, supr_primal)
  RBVars.nₛᵘ = size(RBVars.Φₛᵘ)[2]

  RBVars.Φₛᵘ_quad = hcat(RBVars.Φₛᵘ_quad, supr_primal_quad)
  RBVars.nₛᵘ_quad = size(RBVars.Φₛᵘ_quad)[2]

end

function assemble_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

  RBVars.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
    supr_enrichment_space(RBInfo, RBVars)
  end

  if RBInfo.save_offline_structures
    save_CSV(RBVars.Φₛᵘ, joinpath(RBInfo.ROM_structures_path,"Φₛᵘ.csv"))
    save_CSV(RBVars.Φₛᵖ, joinpath(RBInfo.ROM_structures_path,"Φₛᵖ.csv"))
    save_CSV(RBVars.Φₛᵘ_quad,
      joinpath(RBInfo.ROM_structures_path,"Φₛᵘ_quad.csv"))
  end

  return

end

function get_reduced_basis(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady{T}) where T

  get_reduced_basis(RBInfo, RBVars.Stokes)
  println("Importing the spatial reduced basis for field u, quadrature points")
  RBVars.Φₛᵘ_quad = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₛᵘ_quad.csv"))
  (RBVars.Nₛᵘ_quad, RBVars.nₛᵘ_quad) = size(RBVars.Φₛᵘ_quad)

end

function set_operators(
  RBInfo::Info,
  RBVars::NavierStokesSteady)

  append!(["C"], set_operators(RBInfo, RBVars.Stokes))

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  var::String)

  if var == "C"
    println("The matrix C is non-affine:
      running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
    if isempty(RBVars.MDEIM_mat_C)
      (RBVars.MDEIM_mat_C, RBVars.MDEIM_idx_C, RBVars.MDEIMᵢ_C,
      RBVars.row_idx_C,RBVars.sparse_el_C) = MDEIM_offline(RBInfo, RBVars, "C")
    end
    assemble_reduced_mat_MDEIM(RBVars,RBVars.MDEIM_mat_C,RBVars.row_idx_C)
  else
    assemble_MDEIM_matrices(RBInfo, RBVars.Stokes, var)
  end

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.Stokes, var)

end

function save_M_DEIM_structures(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

  list_M_DEIM = (RBVars.MDEIM_mat_C, RBVars.MDEIMᵢ_C, RBVars.MDEIM_idx_C,
    RBVars.row_idx_C, RBVars.sparse_el_C)
  list_names = ("MDEIM_mat_C","MDEIMᵢ_C","MDEIM_idx_C","row_idx_C","sparse_el_C")

  save_structures_in_list(list_M_DEIM, list_names,
    RBInfo.ROM_structures_path)

end

function get_M_DEIM_structures(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

  operators = String[]
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars.Stokes))

  if "C" ∈ RBInfo.probl_nl

    if isfile(joinpath(RBInfo.ROM_structures_path, "MDEIMᵢ_B.csv"))
      println("Importing MDEIM offline structures, B")
      RBVars.MDEIMᵢ_C = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path,
        "MDEIMᵢ_C.csv"))
      RBVars.MDEIM_idx_C = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.ROM_structures_path,
        "MDEIM_idx_C.csv"))
      RBVars.row_idx_C = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.ROM_structures_path,
        "row_idx_C.csv"))
      RBVars.sparse_el_C = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.ROM_structures_path,
        "sparse_el_C.csv"))
    else
      println("Failed to import MDEIM offline structures,
        C: must build them")
      append!(operators, ["C"])
    end

  end

end

function get_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

  operators = String[]

  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_system_blocks(
  RBInfo::Info,
  RBVars::NavierStokesSteady,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int})

  get_system_blocks(RBInfo, RBVars.Stokes, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::NavierStokesSteady,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  save_system_blocks(RBInfo, RBVars.Stokes, LHS_blocks, RHS_blocks, operators)

end

function get_θᵃ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  Param::SteadyParametricInfo)

  get_θᵃ(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function get_θᵇ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  Param::SteadyParametricInfo)

  get_θᵇ(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function get_θᶜ(
  FEMSpace::SteadyProblem,
  RBVars::NavierStokesSteady,
  Param::SteadyParametricInfo)

  C_μ_sparse = T.(assemble_sparse_mat(FEMSpace, FEMInfo, Param, RBVars.sparse_el_C))
  θᶜ = M_DEIM_online(C_μ_sparse, RBVars.MDEIMᵢ_C, RBVars.MDEIM_idx_C)
  θᶜ::Matrix{T}

end

function get_θᶠʰ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  Param::SteadyParametricInfo)

  get_θᶠʰ(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function get_RB_LHS_blocks(
  RBVars::ADRSteady{T},
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  println("Assembling reduced LHS")

  block₁ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ)
  for q = 1:RBVars.Qᵃ
    block₁ += RBVars.Aₙ[:,:,q] * θᵃ[q]
  end
  for q = 1:RBVars.Qᶜ
    block₁ += RBVars.Cₙ[:,:,q] * θᶜ[q]
  end
  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}
  block₂ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵖ)
  for q = 1:RBVars.Qᵇ
    block₂ += RBVars.Bₙ[:,:,q] * θᵇ[q]
  end
  push!(RBVars.LHSₙ, block₂)::Vector{Matrix{T}}

end

function get_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady,
  Param::SteadyParametricInfo)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    LHS_blocks = [1, 2, 3]
    RHS_blocks = [1]
    operators = get_system_blocks(RBInfo, RBVars, LHS_blocks, RHS_blocks)

    θᵃ, θᵇ, θᶜ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      push!(RBVars.LHSₙ, get_RB_LHS_blocks(RBInfo, RBVars, θᵃ, θᵇ, θᶜ))
    end

    if "RHS" ∈ operators
      if !RBInfo.assemble_parametric_RHS
        push!(RBVars.RHSₙ, get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ))
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
      if "L" ∈ RBInfo.probl_nl
        assemble_RB_lifting(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,LHS_blocks,RHS_blocks,operators)

end

function solve_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady{T},
  RBVars::NavierStokesSteady,
  Param::SteadyParametricInfo) where T

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)
  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")
  RBVars.online_time += @elapsed begin
    xₙ = (vcat(hcat(RBVars.LHSₙ[1], RBVars.LHSₙ[2]),
      hcat(RBVars.LHSₙ[3], zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ))) \
      vcat(RBVars.RHSₙ[1], zeros(T, RBVars.nₛᵖ, 1)))
  end

  RBVars.uₙ = xₙ[1:RBVars.nₛᵘ,:]
  RBVars.pₙ = xₙ[RBVars.nₛᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::NavierStokesSteady)
  reconstruct_FEM_solution(RBVars.Stokes)
end

function offline_phase(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

  if RBInfo.get_snapshots
    get_snapshot_matrix(RBInfo, RBVars)
    get_snapshots_success = true
  else
    get_snapshots_success = false
  end

  if RBInfo.get_offline_structures
    get_reduced_basis(RBInfo, RBVars)
    get_basis_success = true
  else
    get_basis_success = false
  end

  if !get_snapshots_success && !get_basis_success
    error("Impossible to assemble the reduced problem if
      neither the snapshots nor the bases can be loaded")
  end

  if get_snapshots_success && !get_basis_success
    println("Failed to import the reduced basis, building it via POD")
    assemble_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.get_offline_structures
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
  RBVars::NavierStokesSteady{T},
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

    if RBInfo.get_offline_structures
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
