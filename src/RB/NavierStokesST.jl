include("StokesST.jl")
include("NavierStokesS.jl")
include("NavierStokesST_support.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST{T}) where T

  get_snapshot_matrix(RBInfo, RBVars.Stokes)

  println("Importing the snapshot matrix for field u on quadrature points,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵘ_quad = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ_quadp.csv"),
    DataFrame))[:, 1:RBInfo.nₛ*RBVars.Nₜ]
  println("Dimension of velocity snapshot matrix on quadrature points: $(size(Sᵘ_quad))")

end

function PODs_space(
  RBInfo::Info,
  RBVars::NavierStokesST)

  PODs_space(RBInfo, RBVars.Steady)

end

function supr_enrichment_space(
  RBInfo::Info,
  RBVars::NavierStokesST)

  supr_enrichment_space(RBInfo, RBVars.Steady)

end

function PODs_time(
  RBInfo::ROMInfoST,
  RBVars::StokesST{T}) where T

  PODs_time(RBInfo, RBVars.Stokes)

  println("Performing the temporal POD for field u on quadrature points")

  if RBInfo.time_reduction_technique == "ST-HOSVD"
    Sᵘ_quad = RBVars.Φₛᵘ_quad' * RBVars.Sᵘ_quad
  else
    Sᵘ_quad = RBVars.Sᵘ_quad
  end
  Sᵘₜ_quad = mode₂_unfolding(Sᵘ_quad, RBInfo.nₛ)

  RBVars.Φₜᵘ_quad = POD(Sᵘₜ_quad, RBInfo.ϵₜ)
  RBVars.nₜᵘ_quad = size(RBVars.Φₜᵘ_quad)[2]

end

function supr_enrichment_time(
  RBVars::NavierStokesST)

  supr_enrichment_time(RBVars.Stokes)
  RBVars.Φₜᵘ_quad = time_supremizers(RBVars.Φₜᵘ_quad, RBVars.Φₜᵖ)
  RBVars.nₜᵘ_quad = size(RBVars.Φₜᵘ_quad)[2]

end

function assemble_reduced_basis(
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST)

  RBVars.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
    supr_enrichment_space(RBInfo, RBVars.Steady)
    PODs_time(RBInfo, RBVars)
    supr_enrichment_time(RBVars)
  end

  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ
  RBVars.Nᵘ = RBVars.Nₛᵘ * RBVars.Nₜ
  RBVars.nᵖ = RBVars.nₛᵖ * RBVars.nₜᵖ
  RBVars.Nᵖ = RBVars.Nₛᵖ * RBVars.Nₜ
  RBVars.nᵘ_quad = RBVars.nₛᵘ_quad * RBVars.nₜᵘ_quad

  if RBInfo.save_offline_structures
    save_CSV(RBVars.Φₛᵘ, joinpath(RBInfo.ROM_structures_path, "Φₛᵘ.csv"))
    save_CSV(RBVars.Φₜᵘ, joinpath(RBInfo.ROM_structures_path, "Φₜᵘ.csv"))
    save_CSV(RBVars.Φₛᵖ, joinpath(RBInfo.ROM_structures_path, "Φₛᵖ.csv"))
    save_CSV(RBVars.Φₜᵖ, joinpath(RBInfo.ROM_structures_path, "Φₜᵖ.csv"))
    save_CSV(RBVars.Φₛᵘ_quad,
      joinpath(RBInfo.ROM_structures_path, "Φₛᵘ_quad.csv"))
    save_CSV(RBVars.Φₜᵘ_quad,
      joinpath(RBInfo.ROM_structures_path, "Φₜᵘ_quad.csv"))
  end

  return

end

function get_reduced_basis(
  RBInfo::ROMInfoST{T},
  RBVars::NavierStokesST) where T

  get_reduced_basis(RBInfo, RBVars.Stokes)
  println("Importing the space and time reduced basis for field u, quadrature points")
  RBVars.Φₛᵘ_quad = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₛᵘ_quad.csv"))
  RBVars.Φₜᵘ_quad = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₜᵘ_quad.csv"))

  RBVars.nₛᵘ_quad = size(RBVars.Φₛᵘ_quad)[2]
  RBVars.nₜᵘ_quad = size(RBVars.Φₜᵘ_quad)[2]
  RBVars.nᵘ_quad = RBVars.nₛᵘ_quad * RBVars.nₜᵘ_quad

end

function index_mapping(i::Int, j::Int, RBVars::NavierStokesST, var="u")

  index_mapping(i, j, RBVars.Stokes, var)

end

function set_operators(
  RBInfo::Info,
  RBVars::NavierStokesST)

  append!(["M"], set_operators(RBInfo, RBVars.Steady))

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST,
  var::String)

  if var == "C"
    println("The matrix C is non-affine:
      running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
    if isempty(RBVars.MDEIM_mat_C)
      (RBVars.MDEIM_mat_C, RBVars.MDEIM_idx_C, RBVars.MDEIMᵢ_C, RBVars.row_idx_C,
      RBVars.sparse_el_C, RBVars.MDEIM_idx_time_C) = MDEIM_offline(RBInfo, RBVars, "C")
    end
    assemble_reduced_mat_MDEIM(RBVars,RBVars.MDEIM_mat_C,RBVars.row_idx_C)
  else
    assemble_MDEIM_matrices(RBInfo, RBVars.Stokes, var)
  end

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST,
  var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.Stokes, var)

end

function save_M_DEIM_structures(
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST)

  list_M_DEIM = (RBVars.MDEIM_mat_C, RBVars.MDEIMᵢ_C, RBVars.MDEIM_idx_C,
    RBVars.row_idx_C, RBVars.sparse_el_C, RBVars.MDEIM_idx_time_C)
  list_names = ("MDEIM_mat_C","MDEIMᵢ_C","MDEIM_idx_C","row_idx_C",
    "sparse_el_C", "MDEIM_idx_time_C")

  save_structures_in_list(list_M_DEIM, list_names,
    RBInfo.ROM_structures_path)

end

function get_M_DEIM_structures(
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST)

  get_M_DEIM_structures(RBInfo, RBVars.Stokes)

end

function get_offline_structures(
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST)

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_θᵐ(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST,
  Param::ParamInfoST)

  get_θᵐ(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function get_θᵃ(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST,
  Param::ParamInfoST)

  get_θᵃ(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function get_θᵇ(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST,
  Param::ParamInfoST)

  get_θᵇ(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function get_θᶜ(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST,
  Param::ParamInfoST)

  timesθ = get_timesθ(RBInfo)

  if RBInfo.st_M_DEIM
    red_timesθ = timesθ[RBVars.MDEIM_idx_time_C]
    C_μ_sparse = T.(assemble_sparse_mat(
      FEMSpace,FEMInfo,Param,RBVars.sparse_el_C,red_timesθ;var="C"))
    θᶜ = interpolated_θ(RBVars, C_μ_sparse, timesθ, RBVars.MDEIMᵢ_C,
      RBVars.MDEIM_idx_C, RBVars.MDEIM_idx_time_C, RBVars.Qᵐ)
  else
    C_μ_sparse = T.(assemble_sparse_mat(
      FEMSpace,FEMInfo,Param,RBVars.sparse_el_C,timesθ;var="C"))
    θᶜ = (RBVars.MDEIMᵢ_C \
      Matrix{T}(reshape(C_μ_sparse, :, RBVars.Nₜ)[RBVars.MDEIM_idx_C, :]))
  end

  θᶜ::Matrix{T}

end

function get_θᶠʰ(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST,
  Param::ParamInfoST)

  get_θᶠʰ(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function solve_RB_system(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST,
  Param::ParamInfoST)

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

function reconstruct_FEM_solution(RBVars::NavierStokesST)

  reconstruct_FEM_solution(RBVars.Stokes)

end

function offline_phase(
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST)

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

function online_phase(
  RBInfo::ROMInfoST{T},
  RBVars::NavierStokesST,
  param_nbs) where T

  println("Online phase of the RB solver, unsteady Stokes problem")

  μ = load_CSV(Array{T}[],
    joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)

  get_norm_matrix(RBInfo, RBVars.Steady)
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
  path_μ = joinpath(RBInfo.results_path, string_param_nbs)

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

    if RBInfo.get_offline_structures
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
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST{T},
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
    println("Considering parameter number: $nb/$(param_nbs[end])")

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
      RBVars.Stokes, uₕ_test, RBVars.ũ, RBVars.Xᵘ₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test-RBVars.ũ)/length(param_nbs)
    L2_err_nb, L2_L2_err_nb = compute_errors(
      RBVars.Stokes, pₕ_test, RBVars.p̃, RBVars.Xᵖ₀)
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
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::NavierStokesST{T},
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
