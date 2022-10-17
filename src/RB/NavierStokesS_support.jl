################################# OFFLINE ######################################

function set_operators(
  RBInfo::Info,
  RBVars::NavierStokesS)

  append!(["C", "D"], set_operators(RBInfo, RBVars.Stokes))

end

function get_A(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_A(RBInfo, RBVars.Stokes)

end

function get_B(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_B(RBInfo, RBVars.Stokes)

end

function get_C(
  RBInfo::Info,
  RBVars::NavierStokesS{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Cₙ.csv"))

    RBVars.Cₙ = load_CSV(Matrix{T}[],
      joinpath(RBInfo.ROM_structures_path, "Cₙ.csv"))

    (RBVars.MDEIM_C.Matᵢ, RBVars.MDEIM_C.idx, RBVars.MDEIM_C.el) =
      load_structures_in_list(("Matᵢ_C", "idx_C", "el_C"),
      (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
      RBInfo.ROM_structures_path)

  else

    println("Failed to import offline structures for C: must build them")
    op = ["C"]

  end

  op

end

function get_D(
  RBInfo::Info,
  RBVars::NavierStokesS{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Dₙ.csv"))

    RBVars.Dₙ = load_CSV(Matrix{T}[],
      joinpath(RBInfo.ROM_structures_path, "Dₙ.csv"))

    (RBVars.MDEIM_D.Matᵢ, RBVars.MDEIM_D.idx, RBVars.MDEIM_D.el) =
      load_structures_in_list(("Matᵢ_D", "idx_D", "el_D"),
      (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
      RBInfo.ROM_structures_path)

  else

    println("Failed to import offline structures for D: must build them")
    op = ["D"]

  end

  op

end

function get_F(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_F(RBInfo, RBVars.Stokes)

end

function get_H(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_H(RBInfo, RBVars.Stokes)

end

function get_L(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_L(RBInfo, RBVars.Stokes)

end

function get_Lc(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_Lc(RBInfo, RBVars.Stokes)

end

function assemble_affine_structures(
  RBInfo::Info,
  RBVars::NavierStokesS{T},
  var::String) where T

  assemble_affine_structures(RBInfo, RBVars.Stokes, var)

end

function assemble_MDEIM_structures(
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS,
  var::String)

  if var == "C"
    if isempty(RBVars.MDEIM_C.Mat)
      MDEIM_offline!(RBVars.MDEIM_C, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_C, var)
  elseif var == "D"
    if isempty(RBVars.MDEIM_D.Mat)
      MDEIM_offline!(RBVars.MDEIM_D, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_D, var)
  else
    assemble_MDEIM_structures(RBInfo, RBVars.Stokes, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::NavierStokesS{T},
  MDEIM::MMDEIM,
  var::String) where T

  if var ∈ ("C", "D")
    Q = size(MDEIM.Mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, RBVars.Nₛᵘ)

    assemble_VecMatΦ(i) = assemble_ith_row_MatΦ(MDEIM.Mat, RBVars.Φₛ, r_idx, c_idx, i)
    VecMatΦ = Broadcasting(assemble_VecMatΦ)(1:RBVars.Nₛᵘ)::Vector{Matrix{T}}
    MatΦ = Matrix{T}(reduce(vcat, VecMatΦ))::Matrix{T}
    Matₙ = reshape(RBVars.Φₛ' * MatΦ, RBVars.nₛᵘ, :, Q)

    if var == "C"
      RBVars.Cₙ = [Matₙ[:,:,q] for q = 1:Q]
    else
      RBVars.Dₙ = [Matₙ[:,:,q] for q = 1:Q]
    end

  else
    assemble_reduced_mat_MDEIM(RBVars.Stokes, MDEIM, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::NavierStokesS{T},
  MDEIM::VMDEIM,
  var::String) where T

  assemble_reduced_mat_MDEIM(RBVars.Stokes, MDEIM, var)

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::NavierStokesS{T},
  operators::Vector{String}) where T

  affine_vars, affine_names = (RBVars.Cₙ, RBVars.Dₙ), ("Cₙ", "Dₙ")
  affine_entry = get_affine_entries(operators, affine_names)
  save_structures_in_list(affine_vars[affine_entry], affine_names[affine_entry],
    RBInfo.ROM_structures_path)

  MDEIM_vars = (
    RBVars.MDEIM_C.Matᵢ, RBVars.MDEIM_C.idx, RBVars.MDEIM_C.el,
    RBVars.MDEIM_D.Matᵢ, RBVars.MDEIM_D.idx, RBVars.MDEIM_D.el)
  MDEIM_names = (
    "Matᵢ_C","idx_C","el_C",
    "Matᵢ_D","idx_D","el_D")
  save_structures_in_list(MDEIM_vars, MDEIM_names, RBInfo.ROM_structures_path)

  operators_to_pass = setdiff(operators, ("C", "D"))
  save_assembled_structures(RBInfo, RBVars.Stokes, operators_to_pass)

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::Info,
  RBVars::NavierStokesS{T},
  RHS_blocks::Vector{Int}) where T

  if !RBInfo.get_offline_structures
    return ["RHS"]
  end

  operators = String[]

  for i = RHS_blocks
    RHSₙi = "RHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(RBInfo.ROM_structures_path, RHSₙi))
      append!(operators, ["RHS"])
      break
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
  RBVars::NavierStokesS,
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  if ("F" ∉ RBInfo.probl_nl && "H" ∉ RBInfo.probl_nl && "L" ∉ RBInfo.probl_nl
    && "Lc" ∉ RBInfo.probl_nl && "RHS" ∈ operators)
  for i = RHS_blocks
    RHSₙi = "RHSₙ" * string(i) * ".csv"
    save_CSV(RBVars.RHSₙ[i],joinpath(RBInfo.ROM_structures_path, RHSₙi))
  end
end

end

function get_θ_matrix(
  FEMSpace::FOMS,
  RBInfo,
  RBVars::NavierStokesS,
  Param::ParamInfoS,
  var::String) where T

  if var == "C"
    θ_function(FEMSpace, RBVars, RBVars.MDEIM_C, "C")
  elseif var == "D"
    θ_function(FEMSpace, RBVars, RBVars.MDEIM_D, "D")
  else
    get_θ_matrix(FEMSpace, RBInfo, RBVars.Stokes, Param, var)
  end

end

################################################################################

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

  if RBInfo.save_offline
    save_assembled_structures(RBInfo, RBVars, operators)
  end

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
  FEMSpace::FOMS,
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

  Cₙ(u::FEFunction) = sum(Broadcasting(.*)(RBVars.Cₙ, θᶜ(u)))::Matrix{T}
  Dₙ(u::FEFunction) = sum(Broadcasting(.*)(RBVars.Dₙ, θᵈ(u)))::Matrix{T}

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
  FEMSpace::FOMS,
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS,
  Param::ParamInfoS)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
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
  FEMSpace::FOMS,
  RBVars::NavierStokesS{T},
  JinvₙResₙ::Function,
  ϵ=1e-9,
  max_k=10) where T

  x̂mat = zeros(T, RBVars.nₛᵘ + RBVars.nₛᵖ, 1)
  δx̂ = 1. .+ x̂mat
  u = FEFunction(FEMSpace.V[1], zeros(T, RBVars.Nₛᵘ))
  k = 1

  while k ≤ max_k && norm(δx̂) ≥ ϵ
    println("Iter: $k; ||δx̂||₂: $(norm(δx̂))")
    δx̂ = JinvₙResₙ(u, x̂mat)
    x̂mat -= δx̂
    u = FEFunction(FEMSpace.V[1], RBVars.Φₛ * x̂mat[1:RBVars.nₛᵘ])
    k += 1
  end

  println("Newton-Raphson ended with iter: $k; ||δx̂||₂: $(norm(δx̂))")
  x̂mat::Matrix{T}

end

function solve_RB_system(
  FEMSpace::FOMS,
  RBInfo,
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

  FEMSpace, μ = get_FEMμ_info(RBInfo)

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

    Param = ParamInfo(RBInfo, μ[nb])

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

    H1_err_nb = compute_errors(RBVars, uₕ_test, RBVars.ũ, RBVars.X₀[1])
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test - RBVars.ũ) / length(param_nbs)

    L2_err_nb = compute_errors(RBVars, pₕ_test, RBVars.p̃, RBVars.X₀[2])
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
  res_path = joinpath(RBInfo.results_path, string_param_nbs)

  if RBInfo.save_online

    create_dir(res_path)
    save_CSV(ũ_μ, joinpath(res_path, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(res_path, "uₙ.csv"))
    save_CSV(mean_pointwise_err_u, joinpath(res_path, "mean_point_err_u.csv"))
    save_CSV([mean_H1_err], joinpath(res_path, "H1_err.csv"))
    save_CSV(p̃_μ, joinpath(res_path, "p̃.csv"))
    save_CSV(pₙ_μ, joinpath(res_path, "pₙ.csv"))
    save_CSV(mean_pointwise_err_p, joinpath(res_path, "mean_point_err_p.csv"))
    save_CSV([mean_L2_err], joinpath(res_path, "L2_err.csv"))

    if RBInfo.get_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time,"rec_time"=>mean_reconstruction_time)

    CSV.write(joinpath(res_path, "times.csv"),times)

  end

  pass_to_pp = Dict("res_path"=>res_path, "FEMSpace"=>FEMSpace,
    "mean_point_err_u"=>Float.(mean_pointwise_err_u),
    "mean_point_err_p"=>Float.(mean_pointwise_err_p))

  if RBInfo.post_process
    post_process(RBInfo, pass_to_pp)
  end

end
