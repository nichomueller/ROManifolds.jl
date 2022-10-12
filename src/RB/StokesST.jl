include("PoissonST.jl")
include("StokesS.jl")
include("StokesST_support.jl")

################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoST,
  RBVars::StokesST{T}) where T

  get_snapshot_matrix(RBInfo, RBVars.Poisson)

  println("Importing the snapshot matrix for field p,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵖ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
    DataFrame))[:, 1:RBInfo.nₛ*RBVars.Nₜ]
  println("Dimension of pressure snapshot matrix: $(size(Sᵖ))")

  RBVars.Sᵖ = Sᵖ
  RBVars.Nₛᵖ = size(Sᵖ)[1]
  RBVars.Nᵖ = RBVars.Nₛᵖ * RBVars.Nₜ

end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::StokesST)

  get_norm_matrix(RBInfo, RBVars.Poisson)
  get_norm_matrix(RBInfo, RBVars.Steady)

end

function assemble_reduced_basis(
  RBInfo::ROMInfoST,
  RBVars::StokesST)

  RBVars.offline_time += @elapsed begin
    POD_space(RBInfo, RBVars)
    supr_enrichment_space(RBInfo, RBVars)
    POD_time(RBInfo, RBVars)
    supr_enrichment_time(RBVars)
  end

  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ
  RBVars.Nᵘ = RBVars.Nₛᵘ * RBVars.Nₜ
  RBVars.nᵖ = RBVars.nₛᵖ * RBVars.nₜᵖ
  RBVars.Nᵖ = RBVars.Nₛᵖ * RBVars.Nₜ

  if RBInfo.save_offline
    save_CSV(RBVars.Φₛ, joinpath(RBInfo.ROM_structures_path, "Φₛ.csv"))
    save_CSV(RBVars.Φₜᵘ, joinpath(RBInfo.ROM_structures_path, "Φₜᵘ.csv"))
    save_CSV(RBVars.Φₛᵖ, joinpath(RBInfo.ROM_structures_path, "Φₛᵖ.csv"))
    save_CSV(RBVars.Φₜᵖ, joinpath(RBInfo.ROM_structures_path, "Φₜᵖ.csv"))
  end

  return

end

function get_reduced_basis(
  RBInfo,
  RBVars::StokesST) where T

  get_reduced_basis(RBInfo, RBVars.Poisson)

  println("Importing the reduced basis for field p")

  RBVars.Φₛᵖ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₛᵖ.csv"))
  RBVars.nₛᵖ = size(RBVars.Φₛᵖ)[2]
  RBVars.Φₜᵖ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₜᵖ.csv"))
  RBVars.nₜᵖ = size(RBVars.Φₜᵖ)[2]
  RBVars.nᵖ = RBVars.nₛᵖ * RBVars.nₜᵖ

end

function get_offline_structures(
  RBInfo::ROMInfoST,
  RBVars::StokesST)

  operators = get_offline_structures(RBInfo, RBVars.Poisson)

  append!(operators, get_B(RBInfo, RBVars))

  if !RBInfo.online_RHS
    append!(operators, get_Lc(RBInfo, RBVars))
  end

  operators

end

function assemble_offline_structures(
  RBInfo::ROMInfoST,
  RBVars::StokesST,
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
  RBInfo::ROMInfoST,
  RBVars::StokesST)

  println("Offline phase of the RB solver, unsteady Stokes problem")

  RBVars.Nₜ = Int(RBInfo.tₗ / RBInfo.δt)

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
    error("Impossible to assemble the reduced problem if neither
      the snapshots nor the bases can be loaded")
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
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::StokesST{T},
  Param::ParamInfoST) where T

  θᵃ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "A")
  θᵇ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "B")
  θᵐ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "M")

  if !RBInfo.online_RHS
    θᶠ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "F")
    θʰ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "H")
    θˡ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "L")
    θˡᶜ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "Lc")
  else
    θᶠ, θʰ, θˡ, θˡᶜ = (Matrix{T}(undef,0,0), Matrix{T}(undef,0,0),
      Matrix{T}(undef,0,0), Matrix{T}(undef,0,0))
  end

  return θᵃ, θᵇ, θᵐ, θᶠ, θʰ, θˡ, θˡᶜ

end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoST,
  RBVars::StokesST{T},
  θᵐ::Matrix,
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  get_RB_LHS_blocks(RBInfo, RBVars.Poisson, θᵐ, θᵃ)

  Qᵇ = RBVars.Qᵇ
  Φₜᵖᵘ_B = zeros(T, RBVars.nₜᵖ, RBVars.nₜᵘ, Qᵇ)
  Φₜᵖᵘ₁_B = zeros(T, RBVars.nₜᵖ, RBVars.nₜᵘ, Qᵇ)

  @simd for i_t = 1:RBVars.nₜᵖ
    for j_t = 1:RBVars.nₜᵘ
      for q = 1:Qᵇ
        Φₜᵖᵘ_B[i_t,j_t,q] = sum(RBVars.Φₜᵖ[:,i_t].*RBVars.Φₜᵘ[:,j_t].*θᵇ[q,:])
        Φₜᵖᵘ₁_B[i_t,j_t,q] = sum(RBVars.Φₜᵖ[2:end,i_t].*RBVars.Φₜᵘ[1:end-1,j_t].*θᵇ[q,2:end])
      end
    end
  end

  Bₙ_tmp = zeros(T, RBVars.nᵖ, RBVars.nᵘ, Qᵇ)
  Bₙ₁_tmp = zeros(T, RBVars.nᵖ, RBVars.nᵘ, Qᵇ)

  @simd for qᵇ = 1:Qᵇ
    Bₙ_tmp[:,:,qᵇ] = kron(RBVars.Bₙ[:,:,qᵇ], Φₜᵖᵘ_B[:,:,qᵇ])::Matrix{T}
    Bₙ₁_tmp[:,:,qᵇ] = kron(RBVars.Bₙ[:,:,qᵇ], Φₜᵖᵘ₁_B[:,:,qᵇ])::Matrix{T}
  end

  #= Bₙ_blocks = matrix_to_blocks(RBVars.Bₙ, Qᵇ)
  Φₜᵖᵘ_B_blocks = matrix_to_blocks(Φₜᵖᵘ_B, Qᵇ)
  Φₜᵖᵘ₁_B_blocks = matrix_to_blocks(Φₜᵖᵘ₁_B, Qᵇ)
  function modified()
    Bₙ_tmp = Matrix{T}(undef, RBVars.nᵖ, RBVars.nᵘ)
    Bₙ₁_tmp = Matrix{T}(undef, RBVars.nᵖ, RBVars.nᵘ)
    m = Broadcasting(kron)

    Bₙ_tmp = sum(m(Bₙ_blocks, Φₜᵖᵘ_B_blocks))
    Bₙ₁_tmp = sum(m(Bₙ_blocks, Φₜᵖᵘ₁_B_blocks))

    Bₙ_tmp, Bₙ₁_tmp
  end =#

  Bₙ = reshape(sum(Bₙ_tmp, dims=3), RBVars.nᵖ, RBVars.nᵘ)
  Bₙ₁ = reshape(sum(Bₙ_tmp, dims=3), RBVars.nᵖ, RBVars.nᵘ)

  block₂ = - RBInfo.θ*Matrix(Bₙ') - (1-RBInfo.θ)*Matrix(Bₙ₁')
  block₃ = RBInfo.θ*Bₙ + (1-RBInfo.θ)*Bₙ₁

  push!(RBVars.LHSₙ, block₂)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, block₃)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBInfo::Info,
  RBVars::StokesST{T},
  θᶠ::Matrix,
  θʰ::Matrix,
  θˡ::Matrix,
  θˡᶜ::Matrix,) where T

  println("Assembling RHS")

  get_RB_RHS_blocks(RBInfo, RBVars.Poisson, θᶠ, θʰ, θˡ)

  Φₜᵖ_Lc = zeros(T, RBVars.nₜᵖ, RBVars.Qˡᶜ)
  @simd for i_t = 1:RBVars.nₜᵖ
    for q = 1:RBVars.Qˡᶜ
      Φₜᵖ_Lc[i_t, q] = sum(RBVars.Φₜᵖ[:, i_t].*θˡᶜ[q,:])
    end
  end
  block₂ = zeros(T, RBVars.nᵖ, 1)
  @simd for i_s = 1:RBVars.nₛᵖ
    for i_t = 1:RBVars.nₜᵖ
      i_st = index_mapping(i_s, i_t, RBVars, "p")
      block₂[i_st] = - RBVars.Lcₙ[i_s,:]' * Φₜᵖ_Lc[i_t,:]
    end
  end

  push!(RBVars.RHSₙ, block₂)

end

function get_RB_system(
  FEMSpace::FOMST,
  RBInfo::Info,
  RBVars::StokesST,
  Param::ParamInfoST)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  LHS_blocks = [1, 2, 3]
  RHS_blocks = [1, 2]

  RBVars.online_time = @elapsed begin

    operators = get_system_blocks(RBInfo,RBVars,LHS_blocks,RHS_blocks)

    θᵃ, θᵇ, θᵐ, θᶠ, θʰ, θˡ, θˡᶜ  = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ, θᵇ)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_RHS
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ, θˡ, θˡᶜ)
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo, RBVars, LHS_blocks, RHS_blocks, operators)

end

function solve_RB_system(
  FEMSpace::FOMST,
  RBInfo,
  RBVars::StokesST,
  Param::ParamInfoST) where T

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")

  RBVars.online_time += @elapsed begin
    @fastmath xₙ = (vcat(hcat(RBVars.LHSₙ[1], RBVars.LHSₙ[2]),
      hcat(RBVars.LHSₙ[3], zeros(T, RBVars.nᵖ, RBVars.nᵖ))) \
      vcat(RBVars.RHSₙ[1], zeros(T, RBVars.nᵖ, 1)))
  end

  RBVars.uₙ = xₙ[1:RBVars.nᵘ,:]
  RBVars.pₙ = xₙ[RBVars.nᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::StokesST)

  reconstruct_FEM_solution(RBVars.Poisson)

  pₙ = reshape(RBVars.pₙ, (RBVars.nₜᵖ, RBVars.nₛᵖ))
  @fastmath RBVars.p̃ = RBVars.Φₛᵖ * (RBVars.Φₜᵖ * pₙ)'

end

function loop_on_params(
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::StokesST{T},
  μ::Vector{Vector{T}},
  param_nbs) where T

  H1_L2_err = zeros(T, length(param_nbs))
  mean_H1_err = zeros(T, RBVars.Nₜ)
  mean_H1_L2_err = 0.0
  mean_pointwise_err_u = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)

  L2_L2_err = zeros(T, length(param_nbs))
  mean_L2_err = zeros(T, RBVars.Nₜ)
  mean_L2_L2_err = 0.0
  mean_pointwise_err_p = zeros(T, RBVars.Nₛᵖ, RBVars.Nₜ)

  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  ũ_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs)*RBVars.Nₜ)
  uₙ_μ = zeros(T, RBVars.nᵘ, length(param_nbs))
  mean_uₕ_test = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)

  p̃_μ = zeros(T, RBVars.Nₛᵖ, length(param_nbs)*RBVars.Nₜ)
  pₙ_μ = zeros(T, RBVars.nᵖ, length(param_nbs))
  mean_pₕ_test = zeros(T, RBVars.Nₛᵖ, RBVars.Nₜ)

  for (i_nb, nb) in enumerate(param_nbs)
    println("\n")
    println("Considering parameter number: $nb/$(param_nbs[end])")

    Param = ParamInfo(RBInfo, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:,(nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]
    pₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
      DataFrame))[:,(nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]

    mean_uₕ_test += uₕ_test
    mean_pₕ_test += pₕ_test

    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    if i_nb > 1
      mean_online_time = RBVars.online_time/(length(param_nbs)-1)
      mean_reconstruction_time = reconstruction_time/(length(param_nbs)-1)
    end

    H1_err_nb, H1_L2_err_nb = compute_errors(
      RBVars, uₕ_test, RBVars.ũ, RBVars.Xᵘ₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test-RBVars.ũ)/length(param_nbs)

    L2_err_nb, L2_L2_err_nb = compute_errors(
      RBVars, pₕ_test, RBVars.p̃, RBVars.Xᵖ₀)
    L2_L2_err[i_nb] = L2_L2_err_nb
    mean_L2_err += L2_err_nb / length(param_nbs)
    mean_L2_L2_err += L2_L2_err_nb / length(param_nbs)
    mean_pointwise_err_p += abs.(pₕ_test-RBVars.p̃)/length(param_nbs)

    ũ_μ[:, (i_nb-1)*RBVars.Nₜ+1:i_nb*RBVars.Nₜ] = RBVars.ũ
    uₙ_μ[:, i_nb] = RBVars.uₙ
    p̃_μ[:, (i_nb-1)*RBVars.Nₜ+1:i_nb*RBVars.Nₜ] = RBVars.p̃
    pₙ_μ[:, i_nb] = RBVars.pₙ

    println("Online wall time: $(RBVars.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)")
    println("Relative reconstruction L2-L2 error: $L2_L2_err_nb (snapshot number $nb)")
  end

  return (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err_u,mean_H1_err,mean_H1_L2_err,
    H1_L2_err,p̃_μ,pₙ_μ,mean_pₕ_test,mean_pointwise_err_p,mean_L2_err,mean_L2_L2_err,
    L2_L2_err,mean_online_time,mean_reconstruction_time)

end

function online_phase(
  RBInfo,
  RBVars::StokesST,
  param_nbs) where T

  println("Online phase of the RB solver, unsteady Stokes problem")

  FEMSpace, μ = get_FEMμ_info(RBInfo.FEMInfo)

  get_norm_matrix(RBInfo, RBVars)
  (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err_u,mean_H1_err,mean_H1_L2_err,
    H1_L2_err,p̃_μ,pₙ_μ,mean_pₕ_test,mean_pointwise_err_p,mean_L2_err,mean_L2_L2_err,
    L2_L2_err,mean_online_time,mean_reconstruction_time) =
    loop_on_params(FEMSpace, RBInfo, RBVars, μ, param_nbs)

  adapt_time = 0.
  if RBInfo.adaptivity
    adapt_time = @elapsed begin
      (ũ_μ,uₙ_μ,mean_uₕ_test,_,mean_H1_err,mean_H1_L2_err,
        H1_L2_err,p̃_μ,pₙ_μ,mean_pₕ_test,_,mean_L2_err,mean_L2_L2_err,
        L2_L2_err,_,_) =
      adaptive_loop_on_params(FEMSpace, RBInfo, RBVars, mean_uₕ_test,
      mean_pointwise_err_u, mean_pₕ_test, mean_pointwise_err_p, μ, param_nbs)
    end
  end

  string_param_nbs = "params"
  for Param_nb in param_nbs
    string_param_nbs *= "_" * string(Param_nb)
  end
  res_path = joinpath(RBInfo.results_path, string_param_nbs)

  if RBInfo.save_online
    println("Saving the results...")
    create_dir(res_path)

    save_CSV(ũ_μ, joinpath(res_path, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(res_path, "uₙ.csv"))
    save_CSV(mean_pointwise_err_u, joinpath(res_path, "mean_point_err_u.csv"))
    save_CSV(mean_H1_err, joinpath(res_path, "H1_err.csv"))
    save_CSV([mean_H1_L2_err], joinpath(res_path, "H1L2_err.csv"))

    save_CSV(p̃_μ, joinpath(res_path, "p̃.csv"))
    save_CSV(Pₙ_μ, joinpath(res_path, "Pₙ.csv"))
    save_CSV(mean_pointwise_err_p, joinpath(res_path, "mean_point_err_p.csv"))
    save_CSV(mean_L2_err, joinpath(res_path, "L2_err.csv"))
    save_CSV([mean_L2_L2_err], joinpath(res_path, "L2L2_err.csv"))

    if RBInfo.get_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time+adapt_time,"rec_time"=>mean_reconstruction_time)
    CSV.write(joinpath(res_path, "times.csv"),times)
  end

  pass_to_pp = Dict("res_path"=>res_path,
    "FEMSpace"=>FEMSpace, "H1_L2_err"=>H1_L2_err,
    "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>Float.(mean_pointwise_err_u),
    "L2_L2_err"=>L2_L2_err, "mean_L2_err"=>mean_L2_err,
    "mean_point_err_p"=>Float.(mean_pointwise_err_p))

  if RBInfo.post_process
    println("Post-processing the results...")
    post_process(RBInfo, pass_to_pp)
  end

end
