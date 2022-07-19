include("RBPoisson_unsteady.jl")
include("RBStokes_steady.jl")
include("ST-GRB_Stokes.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady{T}) where T

  get_snapshot_matrix(RBInfo, RBVars.P)

  println("Importing the snapshot matrix for field u,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵖ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
    DataFrame))[:, 1:RBInfo.nₛ*RBVars.Nₜ]
  println("Dimension of pressure snapshot matrix: $(size(Sᵖ))")

  RBVars.Sᵖ = Sᵖ
  RBVars.Nₛᵖ = size(Sᵖ)[1]
  RBVars.Nᵖ = RBVars.Nₛᵖ * RBVars.Nₜ

end

function PODs_space(
  RBInfo::Info,
  RBVars::StokesUnsteady)

  PODs_space(RBInfo, RBVars.P)

  println("Performing the spatial POD for field p, using a tolerance of $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars.S)
  RBVars.Φₛᵖ, _ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.Xᵖ₀)
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function supr_enrichment_space(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady)
  #MODIFY#

  supr_primal = primal_supremizers(RBInfo, RBVars.S)
  RBVars.Φₛᵘ = hcat(RBVars.Φₛᵘ, supr_primal)
  RBVars.nₛᵘ = size(RBVars.Φₛᵘ)[2]

end

function PODs_time(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady{T}) where T

  PODs_time(RBInfo, RBVars.P)

  println("Performing the temporal POD for field p, using a tolerance of $(RBInfo.ϵₜ)")

  if RBInfo.time_reduction_technique == "ST-HOSVD"
    Sᵖₜ = zeros(T, RBVars.Nₜ, RBVars.nₛᵖ * RBInfo.nₛ)
    Sᵖ = RBVars.Φₛᵖ' * RBVars.Sᵖ
    @simd for i in 1:RBInfo.nₛ
      Sᵖₜ[:, (i-1)*RBVars.nₛᵖ+1:i*RBVars.nₛᵖ] =
      Sᵖ[:, (i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ]'
    end
  else
    Sᵖₜ = zeros(T, RBVars.Nₜ, RBVars.Nₛᵖ * RBInfo.nₛ)
    Sᵖ = RBVars.Sᵖ
    @simd for i in 1:RBInfo.nₛ
      Sᵖₜ[:, (i-1)*RBVars.Nₛᵖ+1:i*RBVars.Nₛᵖ] =
      transpose(Sᵖ[:, (i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ])
    end
  end

  Φₜᵖ, _ = POD(Sᵖₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵖ = Φₜᵖ
  RBVars.nₜᵖ = size(Φₜᵖ)[2]

end

function time_supremizers(RBVars::StokesUnsteady{T}) where T

  function compute_projection_on_span(
    ξ_new::Vector{T},
    ξ::Matrix{T}) where T

    proj = zeros(T, size(ξ_new))
    for j = 1:size(ξ)[2]
      proj += ξ[:,j] * (ξ_new' * ξ[:,j]) / (ξ[:,j]' * ξ[:,j])
    end

    proj

  end

  println("Checking if primal supremizers in time need to be added")

  ΦₜᵘΦₜᵖ = RBVars.Φₜᵘ' * RBVars.Φₜᵖ
  ξ = zeros(T, size(ΦₜᵘΦₜᵖ))

  for l = 1:size(ΦₜᵘΦₜᵖ)[2]

    if l == 1
      ξ[:,l] = ΦₜᵘΦₜᵖ[:,1]
      enrich = (norm(ξ[:,l]) ≤ 1e-2)
    else
      ξ[:,l] = compute_projection_on_span(ΦₜᵘΦₜᵖ[:, l], ΦₜᵘΦₜᵖ[:, 1:l-1])
      enrich = (norm(ξ[:,l] - ΦₜᵘΦₜᵖ[:,l]) ≤ 1e-2)
    end

    if enrich
      Φₜᵖ_l_on_Φₜᵘ = compute_projection_on_span(RBVars.Φₜᵖ[:, l], RBVars.Φₜᵘ)
      Φₜᵘ_to_add = ((RBVars.Φₜᵖ[:, l] - Φₜᵖ_l_on_Φₜᵘ) /
        norm(RBVars.Φₜᵖ[:, l] - Φₜᵖ_l_on_Φₜᵘ))
      RBVars.Φₜᵘ = hcat(RBVars.Φₜᵘ, Φₜᵘ_to_add)
      ΦₜᵘΦₜᵖ = hcat(ΦₜᵘΦₜᵖ, Φₜᵘ_to_add' * RBVars.Φₜᵖ)
      RBVars.nₜᵘ += 1
    end

  end

end

function build_reduced_basis(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady)

  RBVars.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
    supr_enrichment_space(RBInfo, RBVars)
    PODs_time(RBInfo, RBVars)
    time_supremizers(RBVars)
  end

  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ
  RBVars.Nᵘ = RBVars.Nₛᵘ * RBVars.Nₜ
  RBVars.nᵖ = RBVars.nₛᵖ * RBVars.nₜᵖ
  RBVars.Nᵖ = RBVars.Nₛᵖ * RBVars.Nₜ

  if RBInfo.save_offline_structures
    save_CSV(RBVars.Φₛᵘ, joinpath(RBInfo.Paths.ROM_structures_path, "Φₛᵘ.csv"))
    save_CSV(RBVars.Φₜᵘ, joinpath(RBInfo.Paths.ROM_structures_path, "Φₜᵘ.csv"))
    save_CSV(RBVars.Φₛᵖ, joinpath(RBInfo.Paths.ROM_structures_path, "Φₛᵖ.csv"))
    save_CSV(RBVars.Φₜᵖ, joinpath(RBInfo.Paths.ROM_structures_path, "Φₜᵖ.csv"))
  end

  return

end

function import_reduced_basis(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::StokesUnsteady) where T

  import_reduced_basis(RBInfo, RBVars.P)

  println("Importing the reduced basis for field p")

  RBVars.Φₛᵖ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.Paths.ROM_structures_path, "Φₛᵖ.csv"))
  RBVars.nₛᵖ = size(RBVars.Φₛᵖ)[2]
  RBVars.Φₜᵖ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.Paths.ROM_structures_path, "Φₜᵖ.csv"))
  RBVars.nₜᵖ = size(RBVars.Φₜᵖ)[2]
  RBVars.nᵖ = RBVars.nₛᵖ * RBVars.nₜᵖ

end

function index_mapping(i::Int, j::Int, RBVars::StokesUnsteady, var="u")

  if var == "u"
    return index_mapping(i, j, RBVars.P)
  elseif var == "p"
    return Int((i-1) * RBVars.nₜᵖ + j)
  else
    error("Unrecognized variable")
  end

end

function get_generalized_coordinates(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady{T},
  snaps::Vector{Int}) where T

  if check_norm_matrix(RBVars.S)
    get_norm_matrix(RBInfo, RBVars)
  end

  get_generalized_coordinates(RBInfo, RBVars.P)

  p̂ = zeros(T, RBVars.nᵖ, length(snaps))
  Φₛᵖ_normed = RBVars.Xᵖ₀ * RBVars.Φₛᵖ
  Π = kron(Φₛᵖ_normed, RBVars.Φₜᵘ)::Matrix{T}

  for (i, i_nₛ) = enumerate(snaps)
    println("Assembling generalized coordinate relative to snapshot $(i_nₛ), field p")
    S_i = RBVars.Sᵖ[:, (i_nₛ-1)*RBVars.Nₜ+1:i_nₛ*RBVars.Nₜ]
    p̂[:, i] = sum(Π, dims=2) .* S_i
  end

  RBVars.p̂ = p̂

  if RBInfo.save_offline_structures
    save_CSV(p̂, joinpath(RBInfo.Paths.ROM_structures_path, "p̂.csv"))
  end

end

function test_offline_phase(RBInfo::ROMInfoUnsteady, RBVars::StokesUnsteady)

  get_generalized_coordinates(RBInfo, RBVars, 1)

  uₙ = reshape(RBVars.û, (RBVars.nₜᵘ, RBVars.nₛᵘ))
  u_rec = RBVars.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'
  err = zeros(RBVars.Nₜ)
  for i = 1:RBVars.Nₜ
    err[i] = compute_errors(RBVars.Pᵘ[:, i], u_rec[:, i])
  end

end

function set_operators(
  RBInfo::Info,
  RBVars::StokesUnsteady)

  append!(["B"], set_operators(RBInfo, RBVars.P))

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady,
  var::String)

  assemble_MDEIM_matrices(RBInfo, RBVars.P, var)

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady,
  var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.P, var)

end

function save_M_DEIM_structures(
  ::ROMInfoUnsteady,
  ::StokesUnsteady)

  error("not implemented")

end

function get_M_DEIM_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady)

  get_M_DEIM_structures(RBInfo, RBVars.P)

end

function get_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady)

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_θᵐ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady,
  Param::ParametricInfoUnsteady)

  get_θᵐ(FEMSpace, RBInfo, RBVars.P, Param)

end

function get_θᵃ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady,
  Param::ParametricInfoUnsteady)

  get_θᵃ(FEMSpace, RBInfo, RBVars.P, Param)

end

function get_θᵇ(
  ::UnsteadyProblem,
  ::ROMInfoUnsteady,
  ::StokesUnsteady{T},
  ::ParametricInfoUnsteady) where T

  reshape([one(T)],1,1)::Matrix{T}

end

function get_θᶠʰ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady,
  Param::ParametricInfoUnsteady)

  get_θᶠʰ(FEMSpace, RBInfo, RBVars.P, Param)

end

function solve_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady,
  Param::ParametricInfoUnsteady)

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")

  RBVars.online_time += @elapsed begin
    @fastmath xₙ = (vcat(hcat(RBVars.LHSₙ[1], RBVars.LHSₙ[2]),
      hcat(RBVars.LHSₙ[3], zeros(T, RBVars.nᵖ, RBVars.nᵖ))) \
      vcat(RBVars.RHSₙ[1], zeros(T, RBVars.nᵖ, 1)))
  end

  RBVars.uₙ = xₙ[1:RBVars.nᵘ,:]
  RBVars.pₙ = xₙ[RBVars.nᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::StokesUnsteady)

  reconstruct_FEM_solution(RBVars.P)

  pₙ = reshape(RBVars.pₙ, (RBVars.nₜᵖ, RBVars.nₛᵖ))
  @fastmath RBVars.p̃ = RBVars.Φₛᵖ * (RBVars.Φₜᵖ * pₙ)'

end

function offline_phase(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady)

  println("Offline phase of the RB solver, unsteady Stokes problem")

  RBVars.Nₜ = Int(RBInfo.tₗ / RBInfo.δt)

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
    error("Impossible to assemble the reduced problem if neither
      the snapshots nor the bases can be loaded")
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
  RBInfo::ROMInfoUnsteady{T},
  RBVars::StokesUnsteady,
  param_nbs) where T

  println("Online phase of the RB solver, unsteady Stokes problem")

  μ = load_CSV(Array{T}[],
    joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)

  get_norm_matrix(RBInfo, RBVars.S)
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
  path_μ = joinpath(RBInfo.Paths.results_path, string_param_nbs)

  if RBInfo.save_results
    println("Saving the results...")
    create_dir(path_μ)

    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err_U, joinpath(path_μ, "mean_point_err_u.csv"))
    save_CSV(mean_H1_err, joinpath(path_μ, "H1_err.csv"))
    save_CSV([mean_H1_L2_err], joinpath(path_μ, "H1L2_err.csv"))

    save_CSV(p̃_μ, joinpath(path_μ, "p̃.csv"))
    save_CSV(Pₙ_μ, joinpath(path_μ, "Pₙ.csv"))
    save_CSV(mean_pointwise_err_p, joinpath(path_μ, "mean_point_err_p.csv"))
    save_CSV(mean_L2_err, joinpath(path_μ, "L2_err.csv"))
    save_CSV([mean_L2_L2_err], joinpath(path_μ, "L2L2_err.csv"))

    if RBInfo.import_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time+adapt_time,"rec_time"=>mean_reconstruction_time)
    CSV.write(joinpath(path_μ, "times.csv"),times)
  end

  pass_to_pp = Dict("path_μ"=>path_μ,
    "FEMSpace"=>FEMSpace, "H1_L2_err"=>H1_L2_err,
    "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>Float.(mean_pointwise_err_u),
    "L2_L2_err"=>L2_L2_err, "mean_L2_err"=>mean_L2_err,
    "mean_point_err_p"=>Float.(mean_pointwise_err_p))

  if RBInfo.post_process
    println("Post-processing the results...")
    post_process(RBInfo, pass_to_pp)
  end

end

function loop_on_params(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady{T},
  μ::Vector{Vector{T}},
  param_nbs) where T

  H1_L2_err = zeros(T, length(param_nbs))
  L2_L2_err = zeros(T, length(param_nbs))
  mean_H1_err = zeros(T, RBVars.Nₜ)
  mean_L2_err = zeros(T, RBVars.Nₜ)
  mean_H1_L2_err = 0.0
  mean_L2_L2_err = 0.0
  mean_pointwise_err_u = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)
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
    println("Considering Parameter number: $nb/$(param_nbs[end])")

    Param = get_ParamInfo(RBInfo, μ[nb])

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
      RBVars.P, uₕ_test, RBVars.ũ, RBVars.Xᵘ₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test-RBVars.ũ)/length(param_nbs)
    L2_err_nb, L2_L2_err_nb = compute_errors(
      RBVars.P, pₕ_test, RBVars.p̃, RBVars.Xᵖ₀)
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

function adaptive_loop_on_params(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesUnsteady{T},
  mean_uₕ_test::Matrix,
  mean_pointwise_err_u::Matrix,
  mean_pₕ_test::Matrix,
  mean_pointwise_err_p::Matrix,
  μ::Vector{Vector{T}},
  param_nbs,
  n_adaptive=nothing) where T

  if isnothing(n_adaptive)
    nₛᵘ_add = floor(Int,RBVars.nₛᵘ*0.1)
    nₜᵘ_add = floor(Int,RBVars.nₜᵘ*0.1)
    n_adaptive_u = maximum(hcat([1,1],[nₛᵘ_add,nₜᵘ_add]),dims=2)::Vector{Int}
    nₛᵖ_add = floor(Int,RBVars.nₛᵖ*0.1)
    nₜᵖ_add = floor(Int,RBVars.nₜᵖ*0.1)
    n_adaptive_p = maximum(hcat([1,1],[nₛᵖ_add,nₜᵖ_add]),dims=2)::Vector{Int}
  end

  println("Running adaptive cycle: adding $n_adaptive_u temporal and spatial bases
    for u, and $n_adaptive_p temporal and spatial bases for p")

  time_err_u = zeros(T, RBVars.Nₜ)
  space_err_u = zeros(T, RBVars.Nₛᵘ)
  time_err_p = zeros(T, RBVars.Nₜ)
  space_err_p = zeros(T, RBVars.Nₛᵖ)
  for iₜ = 1:RBVars.Nₜ
    time_err_u[iₜ] = (mynorm(mean_pointwise_err_u[:,iₜ],RBVars.Xᵘ₀) /
      mynorm(mean_uₕ_test[:,iₜ],RBVars.Xᵘ₀))
    time_err_p[iₜ] = (mynorm(mean_pointwise_err_p[:,iₜ],RBVars.Xᵖ₀) /
      mynorm(mean_pₕ_test[:,iₜ],RBVars.Xᵖ₀))
  end
  for iₛ = 1:RBVars.Nₛᵘ
    space_err_u[iₛ] = mynorm(mean_pointwise_err_u[iₛ,:])/mynorm(mean_uₕ_test[iₛ,:])
  end
  for iₛ = 1:RBVars.Nₛᵖ
    space_err_p[iₛ] = mynorm(mean_pointwise_err_p[iₛ,:])/mynorm(mean_pₕ_test[iₛ,:])
  end

  ind_s_u = argmax(space_err_u,n_adaptive_u[1])
  ind_t_u = argmax(time_err_u,n_adaptive_u[2])
  ind_s_p = argmax(space_err_p,n_adaptive_p[1])
  ind_t_p = argmax(time_err_p,n_adaptive_p[2])

  if isempty(RBVars.Pᵘ)
    Sᵘ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:,1:RBInfo.nₛ*RBVars.Nₜ]
    Sᵖ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
      DataFrame))[:,1:RBInfo.nₛ*RBVars.Nₜ]
  else
    Sᵘ = RBVars.Pᵘ
    Sᵖ = RBVars.Sᵖ
  end
  Sᵘ = reshape(sum(reshape(Sᵘ,RBVars.Nₛᵘ,RBVars.Nₜ,:),dims=3),RBVars.Nₛᵘ,:)
  Sᵖ = reshape(sum(reshape(Sᵖ,RBVars.Nₛᵖ,RBVars.Nₜ,:),dims=3),RBVars.Nₛᵖ,:)

  Φₛᵘ_new = Matrix{T}(qr(Sᵘ[:,ind_t_u]).Q)[:,1:n_adaptive_u[2]]
  Φₜᵘ_new = Matrix{T}(qr(Sᵘ[ind_s_u,:]').Q)[:,1:n_adaptive_u[1]]
  RBVars.nₛᵘ += n_adaptive_u[2]
  RBVars.nₜᵘ += n_adaptive_u[1]
  RBVars.nᵘ = RBVars.nₛᵘ*RBVars.nₜᵘ
  RBVars.Φₛᵘ = Matrix{T}(qr(hcat(RBVars.Φₛᵘ,Φₛᵘ_new)).Q)[:,1:RBVars.nₛᵘ]
  RBVars.Φₜᵘ = Matrix{T}(qr(hcat(RBVars.Φₜᵘ,Φₜᵘ_new)).Q)[:,1:RBVars.nₜᵘ]

  Φₛᵖ_new = Matrix{T}(qr(Sᵖ[:,ind_t_p]).Q)[:,1:n_adaptive_p[2]]
  Φₜᵖ_new = Matrix{T}(qr(Sᵖ[ind_s_p,:]').Q)[:,1:n_adaptive_p[1]]
  RBVars.nₛᵖ += n_adaptive_p[2]
  RBVars.nₜᵖ += n_adaptive_p[1]
  RBVars.nᵖ = RBVars.nₛᵖ*RBVars.nₜᵖ
  RBVars.Φₛᵖ = Matrix{T}(qr(hcat(RBVars.Φₛᵖ,Φₛᵖ_new)).Q)[:,1:RBVars.nₛᵖ]
  RBVars.Φₜᵖ = Matrix{T}(qr(hcat(RBVars.Φₜᵖ,Φₜᵖ_new)).Q)[:,1:RBVars.nₜᵖ]

  RBInfo.save_offline_structures = false
  assemble_offline_structures(RBInfo, RBVars)

  loop_on_params(FEMSpace,RBInfo,RBVars,μ,param_nbs)

end
