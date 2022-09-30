include("StokesS.jl")
include("NavierStokesS_support.jl")

################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS)

  get_snapshot_matrix(RBInfo, RBVars.Stokes)

end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_norm_matrix(RBInfo, RBVars.Stokes)

end

function assemble_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS)

  assemble_reduced_basis(RBInfo, RBVars.Stokes)

end

function get_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS)

  get_reduced_basis(RBInfo, RBVars.Stokes)

end

function get_offline_structures(
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS)

  operators = String[]

  append!(operators, get_A(RBInfo, RBVars))
  append!(operators, get_B(RBInfo, RBVars))
  append!(operators, get_C(RBInfo, RBVars))
  append!(operators, get_D(RBInfo, RBVars))

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
  RBVars::NavierStokesS,
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
  RBVars::NavierStokesS)

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
  RBVars::NavierStokesS,
  Param::ParamInfoS)

  θᶜ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "C")
  θᵈ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "D")
  get_θ(FEMSpace, RBInfo, RBVars.Stokes, Param)..., θᶜ, θᵈ

end

function get_RB_JinvRes(
  RBVars::NavierStokesS{T},
  θᶜ::Function,
  θᵈ::Function) where T

  Cₙ(u::FEFunction) = sum([RBVars.Cₙ[:, :, q] * θᶜ(u)[q] for q = 1:RBVars.Qᶜ])::Matrix{T}
  Dₙ(u::FEFunction) = sum([RBVars.Dₙ[:, :, q] * θᵈ(u)[q] for q = 1:RBVars.Qᵈ])::Matrix{T}

  function JinvₙResₙ(u::FEFunction, x̂::Matrix{T}) where T
    Cₙu, Dₙu = Cₙ(u), Dₙ(u)
    LHSₙ = (vcat(hcat(RBVars.LHSₙ[1] + Cₙu, RBVars.LHSₙ[2]),
      hcat(RBVars.LHSₙ[3], zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ))))::Matrix{T}
    RHSₙ =  vcat(RBVars.RHSₙ[1], zeros(T, RBVars.nₛᵖ, 1))::Matrix{T}
    Jₙ = LHSₙ + (vcat(hcat(Dₙu, zeros(T, RBVars.nₛᵘ, RBVars.nₛᵖ)),
      hcat(zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ), zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ))))::Matrix{T}
    resₙ = (LHSₙ * x̂ - RHSₙ)::Matrix{T}
    (Jₙ \ resₙ)::Matrix{T}
  end

  JinvₙResₙ::Function

end

function get_RB_system(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS,
  Param::ParamInfoS)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  get_Q(RBVars)
  RHS_blocks = [1, 2]

  RBVars.online_time = @elapsed begin
    operators = get_system_blocks(RBInfo, RBVars, RHS_blocks)

    θᵃ, θᵇ, θᶠ, θʰ, θˡ, θˡᶜ, θᶜ, θᵈ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    get_RB_LHS_blocks(RBVars.Stokes, θᵃ, θᵇ)
    if "RHS" ∈ operators
      if !RBInfo.online_RHS
        get_RB_RHS_blocks(RBVars.Stokes, θᶠ, θʰ, θˡ, θˡᶜ)
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars.Stokes, Param)
      end
    end

    JinvₙResₙ = get_RB_JinvRes(RBVars, θᶜ, θᵈ)
  end

  save_system_blocks(RBInfo,RBVars,RHS_blocks,operators)

  JinvₙResₙ::Function

end

function newton(
  FEMSpace::FEMProblemS,
  RBVars::NavierStokesS{T},
  JinvₙResₙ::Function,
  ϵ=1e-9,
  max_k=10) where T

  x̂mat = zeros(T, RBVars.nₛᵘ + RBVars.nₛᵖ, 1)
  δx̂ = 1. .+ x̂mat
  u = FEFunction(FEMSpace.V, zeros(T, RBVars.Nₛᵘ))
  k = 1

  while k ≤ max_k && norm(δx̂) ≥ ϵ
    println("Iter: $k; ||δx̂||₂: $(norm(δx̂))")
    δx̂ = JinvₙResₙ(u, x̂mat)
    x̂mat -= δx̂
    u = FEFunction(FEMSpace.V, RBVars.Φₛᵘ * x̂mat[1:RBVars.nₛᵘ])
    k += 1
  end

  println("Newton-Raphson ended with iter: $k; ||δx̂||₂: $(norm(δx̂))")
  x̂mat::Matrix{T}

end

function solve_RB_system(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::NavierStokesS,
  Param::ParamInfoS) where T

  JinvₙResₙ = get_RB_system(FEMSpace, RBInfo, RBVars, Param)
  println("Solving RB problem via Newton-Raphson iterations")
  RBVars.online_time += @elapsed begin
    xₙ = newton(FEMSpace, RBVars, JinvₙResₙ)
  end

  RBVars.uₙ = xₙ[1:RBVars.nₛᵘ,:]
  RBVars.pₙ = xₙ[RBVars.nₛᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::NavierStokesS)
  reconstruct_FEM_solution(RBVars.Stokes)
end

function online_phase(
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS{T},
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
