include("PoissonS.jl")
include("StokesS_support.jl")

################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoS,
  RBVars::StokesS{T}) where T

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
  RBVars::StokesS{T}) where T

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

function assemble_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::StokesS)

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

function get_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::StokesS{T}) where T

  get_reduced_basis(RBInfo, RBVars.Poisson)

  println("Importing the spatial reduced basis for field p")
  RBVars.Φₛᵖ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₛᵖ.csv"))
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function get_offline_structures(
  RBInfo::ROMInfoS,
  RBVars::StokesS)

  operators = String[]

  append!(operators, get_A(RBInfo, RBVars))
  append!(operators, get_B(RBInfo, RBVars))

  if !RBInfo.online_RHS
    append!(operators, get_F(RBInfo, RBVars))
    append!(operators, get_H(RBInfo, RBVars))
    append!(operators, get_L(RBInfo, RBVars))
    append!(operators, get_Lc(RBInfo, RBVars))
  end

  operators

end

function assemble_offline_structures(
  RBInfo::ROMInfoS,
  RBVars::StokesS,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.offline_time += @elapsed begin
    for var ∈ setdiff(operators, RBInfo.probl_nl)
      assemble_affine_structures(RBInfo, RBVars, var)
    end

    for var ∈ intersect(operators, RBInfo.probl_nl)
      assemble_MDEIM_structures(RBInfo, RBVars, var)
    end
  end

  save_assembled_structures(RBInfo, RBVars, operators)

end

function offline_phase(
  RBInfo::ROMInfoS,
  RBVars::StokesS)

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

################################## ONLINE ######################################

function get_θ(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::StokesS{T},
  Param::ParamInfoS) where T

  θᵃ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "A")
  θᵇ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "B")

  if !RBInfo.online_RHS
    θᶠ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "F")
    θʰ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "H")
    θˡ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "L")
    θˡᶜ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "Lc")
  else
    θᶠ, θʰ, θˡ, θˡᶜ = (Matrix{T}(undef,0,0), Matrix{T}(undef,0,0),
      Matrix{T}(undef,0,0), Matrix{T}(undef,0,0))
  end

  return θᵃ, θᵇ, θᶠ, θʰ, θˡ, θˡᶜ

end

function get_RB_LHS_blocks(
  RBVars::StokesS{T},
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  get_RB_LHS_blocks(RBVars.Poisson, θᵃ)

  block₂ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ)
  for q = 1:RBVars.Qᵇ
    block₂ += RBVars.Bₙ[:,:,q] * θᵇ[q]
  end

  push!(RBVars.LHSₙ, -block₂')::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, block₂)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBVars::StokesS{T},
  θᶠ::Matrix,
  θʰ::Matrix,
  θˡ::Matrix,
  θˡᶜ::Matrix) where T

  get_RB_RHS_blocks(RBVars.Poisson, θᶠ, θʰ, θˡ)
  push!(RBVars.RHSₙ, -RBVars.Lcₙ * θˡᶜ)::Vector{Matrix{T}}

end

function get_RB_system(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::StokesS,
  Param::ParamInfoS)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  get_Q(RBVars)
  LHS_blocks = [1, 2, 3]
  RHS_blocks = [1, 2]

  RBVars.online_time = @elapsed begin
    operators = get_system_blocks(RBInfo, RBVars, LHS_blocks, RHS_blocks)

    θᵃ, θᵇ, θᶠ, θʰ, θˡ, θˡᶜ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBVars, θᵃ, θᵇ)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_RHS
        get_RB_RHS_blocks(RBVars, θᶠ, θʰ, θˡ, θˡᶜ)
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,LHS_blocks,RHS_blocks,operators)

end

function solve_RB_system(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::StokesS,
  Param::ParamInfoS) where T

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)
  println("Solving RB problem via backslash")
  RBVars.online_time += @elapsed begin
    LHSₙ = vcat(hcat(RBVars.LHSₙ[1], RBVars.LHSₙ[2]),
      hcat(RBVars.LHSₙ[3], zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ)))
    RHSₙ = vcat(RBVars.RHSₙ[1], RBVars.RHSₙ[2])
    xₙ = LHSₙ \ RHSₙ
  end
  println("Condition number of the system's matrix: $(cond(LHSₙ))")

  RBVars.uₙ = xₙ[1:RBVars.nₛᵘ,:]
  RBVars.pₙ = xₙ[RBVars.nₛᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::StokesS)
  reconstruct_FEM_solution(RBVars.Poisson)
  RBVars.p̃ = RBVars.Φₛᵖ * RBVars.pₙ
end

function online_phase(
  RBInfo::ROMInfoS,
  RBVars::StokesS{T},
  param_nbs) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)

  mean_H1_err = 0.0
  mean_L2_err = 0.0
  mean_pointwise_err_u = zeros(T, RBVars.Nₛᵘ)
  mean_pointwise_err_p = zeros(T, RBVars.Nₛᵖ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(RBInfo, RBVars)

  ũ_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs))
  uₙ_μ = zeros(T, RBVars.nₛᵘ, length(param_nbs))
  p̃_μ = zeros(T, RBVars.Nₛᵖ, length(param_nbs))
  pₙ_μ = zeros(T, RBVars.nₛᵖ, length(param_nbs))

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

    H1_err_nb = compute_errors(RBVars, uₕ_test, RBVars.ũ, RBVars.Xᵘ₀)
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test - RBVars.ũ) / length(param_nbs)

    L2_err_nb = compute_errors(RBVars, pₕ_test, RBVars.p̃, RBVars.Xᵖ₀)
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
