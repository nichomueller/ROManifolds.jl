include("RBADR_steady.jl")
include("ST-GRB_ADR.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady{T}) where T

  get_snapshot_matrix(RBInfo, RBVars.Poisson)

end

PODs_space(RBInfo::Info, RBVars::ADRUnsteady) =
  PODs_space(RBInfo, RBVars.Steady)

function PODs_time(
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady{T}) where T

  PODs_time(RBInfo, RBVars.Poisson)

end

function build_reduced_basis(
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady)

  build_reduced_basis(RBInfo, RBVars.Poisson)

end

function import_reduced_basis(
  RBInfo::Info,
  RBVars::ADRUnsteady{T}) where T

  import_reduced_basis(RBInfo, RBVars.Poisson)

end

function index_mapping(
  i::Int,
  j::Int,
  RBVars::ADRUnsteady)

  index_mapping(i, j, RBVars.Poisson)

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady,
  var::String)

  if var == "B"
    println("The matrix $var is non-affine:
      running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
    if isempty(RBVars.MDEIM_mat_B)
      (RBVars.MDEIM_mat_B, RBVars.MDEIM_idx_B, RBVars.MDEIMᵢ_B, RBVars.row_idx_B,
        RBVars.sparse_el_B, RBVars.MDEIM_idx_time_B) = MDEIM_offline(RBInfo, RBVars, "B")
    end
    assemble_reduced_mat_MDEIM(
      RBVars,RBVars.MDEIM_mat_B,RBVars.row_idx_B,"B")
  elseif var == "D"
    if isempty(RBVars.MDEIM_mat_D)
      (RBVars.MDEIM_mat_D, RBVars.MDEIM_idx_D, RBVars.MDEIMᵢ_D,
      RBVars.row_idx_D,RBVars.sparse_el_D, RBVars.MDEIM_idx_time_D) = MDEIM_offline(RBInfo, RBVars, "D")
    end
    assemble_reduced_mat_MDEIM(
      RBVars,RBVars.MDEIM_mat_D,RBVars.row_idx_D,"D")
  else
    assemble_MDEIM_matrices(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady,
  var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.Poisson, var)

end

function save_M_DEIM_structures(
  RBInfo::Info,
  RBVars::ADRUnsteady)

  list_M_DEIM = (RBVars.MDEIM_mat_B, RBVars.MDEIMᵢ_B, RBVars.MDEIM_idx_B,
    RBVars.sparse_el_B, RBVars.row_idx_B, RBVars.MDEIM_idx_time_B,
    RBVars.MDEIM_mat_D, RBVars.MDEIMᵢ_D, RBVars.MDEIM_idx_D,
    RBVars.sparse_el_D, RBVars.row_idx_D, RBVars.MDEIM_idx_time_D)
  list_names = ("MDEIM_mat_B", "MDEIMᵢ_B", "MDEIM_idx_B", "sparse_el_B",
   "row_idx_B", "MDEIM_idx_time_B", "MDEIM_mat_D", "MDEIMᵢ_D", "MDEIM_idx_D",
   "sparse_el_D", "row_idx_D", "MDEIM_idx_time_D")

  save_structures_in_list(list_M_DEIM, list_names,
    RBInfo.ROM_structures_path)

end

function set_operators(
  RBInfo::Info,
  RBVars::ADRUnsteady)

  vcat(["M"], set_operators(RBInfo, RBVars.Steady))

end

function get_M_DEIM_structures(
  RBInfo::Info,
  RBVars::ADRUnsteady{T}) where T

  operators = String[]

  if "A" ∈ RBInfo.probl_nl
    if isfile(joinpath(RBInfo.ROM_structures_path, "MDEIM_idx_time_A.csv"))
      RBVars.MDEIM_idx_time_A = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.ROM_structures_path, "MDEIM_idx_time_A.csv"))
    else
      append!(operators, ["A"])
    end
  end

  if "B" ∈ RBInfo.probl_nl
    if isfile(joinpath(RBInfo.ROM_structures_path, "MDEIM_idx_time_B.csv"))
      RBVars.MDEIM_idx_time_B = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.ROM_structures_path, "MDEIM_idx_time_B.csv"))
    else
      append!(operators, ["B"])
    end
  end

  if "D" ∈ RBInfo.probl_nl
    if isfile(joinpath(RBInfo.ROM_structures_path, "MDEIM_idx_time_D.csv"))
      RBVars.MDEIM_idx_time_D = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.ROM_structures_path, "MDEIM_idx_time_D.csv"))
    else
      append!(operators, ["D"])
    end
  end

  if "M" ∈ RBInfo.probl_nl

    if isfile(joinpath(RBInfo.ROM_structures_path, "MDEIMᵢ_M.csv"))
      println("Importing MDEIM offline structures for the mass matrix")
      RBVars.MDEIMᵢ_M = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path,
        "MDEIMᵢ_M.csv"))
      RBVars.MDEIM_idx_M = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.ROM_structures_path,
        "MDEIM_idx_M.csv"))
      RBVars.sparse_el_M = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.ROM_structures_path,
        "sparse_el_M.csv"))
      RBVars.row_idx_M = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.ROM_structures_path,
        "row_idx_M.csv"))
      RBVars.MDEIM_idx_time_M = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.ROM_structures_path, "MDEIM_idx_time_M.csv"))
      append!(operators, [])
    else
      println("Failed to import MDEIM offline structures for the mass matrix: must build them")
      append!(operators, ["M"])
    end

  end

  if "F" ∈ RBInfo.probl_nl
    if isfile(joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_F.csv"))
    RBVars.DEIM_idx_time_F = load_CSV(Vector{Int}(undef,0),
      joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_F.csv"))
    else
      append!(operators, ["F"])
    end
  end

  if "H" ∈ RBInfo.probl_nl
    if isfile(joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_H.csv"))
    RBVars.DEIM_idx_time_H = load_CSV(Vector{Int}(undef,0),
      joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_H.csv"))
    else
      append!(operators, ["H"])
    end
  end

  append!(operators, get_M_DEIM_structures(RBInfo, RBVars.Steady))

end

function get_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady)

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_θᵐ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady{T},
  Param::UnsteadyParametricInfo) where T

  get_θᵐ(FEMSpace, RBInfo, RBVars, Param)

end

function get_θᵃ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady{T},
  Param::UnsteadyParametricInfo) where T

  get_θᵃ(FEMSpace, RBInfo, RBVars, Param)

end

function get_θᵇ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady{T},
  Param::UnsteadyParametricInfo) where T

  timesθ = get_timesθ(RBInfo)

  if "B" ∉ RBInfo.probl_nl
    θᵇ = T.(zeros(T, 1, RBVars.Nₜ))
    for (i_t, t_θ) = enumerate(timesθ)
      θᵇ[i_t] = Param.bₜ(t_θ)
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[RBVars.MDEIM_idx_time_B]
      B_μ_sparse = T.(build_sparse_mat(
        FEMSpace,FEMInfo,Param,RBVars.sparse_el_B,red_timesθ;var="B"))
      θᵇ = interpolated_θ(RBVars, B_μ_sparse, timesθ, RBVars.MDEIMᵢ_B,
        RBVars.MDEIM_idx_B, RBVars.MDEIM_idx_time_B, RBVars.Qᵐ)
    else
      B_μ_sparse = T.(build_sparse_mat(
        FEMSpace,FEMInfo,Param,RBVars.sparse_el_B,timesθ;var="B"))
      θᵇ = (RBVars.MDEIMᵢ_B \
        Matrix{T}(reshape(B_μ_sparse, :, RBVars.Nₜ)[RBVars.MDEIM_idx_B, :]))
    end
  end

  θᵇ::Matrix{T}

end

function get_θᵈ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady{T},
  Param::UnsteadyParametricInfo) where T

  timesθ = get_timesθ(RBInfo)

  if !"D" ∈ RBInfo.probl_nl
    θᵈ = T.(zeros(T, 1, RBVars.Nₜ))
    for (i_t, t_θ) = enumerate(timesθ)
      θᵈ[i_t] = Param.σₜ(t_θ)
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[RBVars.MDEIM_idx_time_D]
      D_μ_sparse = T.(build_sparse_mat(
        FEMSpace,FEMInfo,Param,RBVars.sparse_el_D,red_timesθ;var="D"))
      θᵈ = interpolated_θ(RBVars, D_μ_sparse, timesθ, RBVars.MDEIMᵢ_D,
        RBVars.MDEIM_idx_D, RBVars.MDEIM_idx_time_D, RBVars.Qᵈ)
    else
      D_μ_sparse = T.(build_sparse_mat(
        FEMSpace,FEMInfo,Param,RBVars.sparse_el_D,timesθ;var="D"))
      θᵈ = (RBVars.MDEIMᵢ_D \
        Matrix{T}(reshape(D_μ_sparse, :, RBVars.Nₜ)[RBVars.MDEIM_idx_D, :]))
    end
  end

  θᵈ::Matrix{T}

end

function get_θᶠʰ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady{T},
  Param::UnsteadyParametricInfo) where T

  get_θᶠʰ(FEMSpace, RBInfo, RBVars, Param)

end

function solve_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady,
  Param::UnsteadyParametricInfo)

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")

  RBVars.online_time += @elapsed begin
    @fastmath RBVars.uₙ = RBVars.LHSₙ[1] \ RBVars.RHSₙ[1]
  end

end

function reconstruct_FEM_solution(RBVars::ADRUnsteady)

  reconstruct_FEM_solution(RBVars.Poisson)

end

function offline_phase(
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady)

  println("Offline phase of the RB solver, unsteady ADR problem")

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
  RBVars::ADRUnsteady,
  param_nbs) where T

  println("Online phase of the RB solver, unsteady ADR problem")

  μ = load_CSV(Array{T}[],
    joinpath(get_FEM_snap_path(RBInfo), "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(get_mesh_path(RBInfo))
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)

  get_norm_matrix(RBInfo, RBVars.Steady)
  (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,H1_L2_err,
    mean_online_time,mean_reconstruction_time) =
    loop_on_params(FEMSpace, RBInfo, RBVars, μ, param_nbs)

  adapt_time = 0.
  if RBInfo.adaptivity
    adapt_time = @elapsed begin
      (ũ_μ,uₙ_μ,_,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,
      H1_L2_err,_,_) =
      adaptive_loop_on_params(FEMSpace, RBInfo, RBVars, mean_uₕ_test,
      mean_pointwise_err, μ, param_nbs)
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
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV(mean_H1_err, joinpath(path_μ, "H1_err.csv"))
    save_CSV(H1_L2_err, joinpath(path_μ, "H1L2_err.csv"))

    if RBInfo.import_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time+adapt_time,"rec_time"=>mean_reconstruction_time)
    CSV.write(joinpath(path_μ, "times.csv"),times)
  end

  pass_to_pp = Dict("path_μ"=>path_μ,
    "FEMSpace"=>FEMSpace, "H1_L2_err"=>H1_L2_err,
    "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>Float.(mean_pointwise_err))

  if RBInfo.post_process
    println("Post-processing the results...")
    post_process(RBInfo, pass_to_pp)
  end

end

function loop_on_params(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady{T},
  μ::Vector{Vector{T}},
  param_nbs) where T

  H1_L2_err = zeros(T, length(param_nbs))
  mean_H1_err = zeros(T, RBVars.Nₜ)
  mean_H1_L2_err = 0.0
  mean_pointwise_err = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  ũ_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs)*RBVars.Nₜ)
  uₙ_μ = zeros(T, RBVars.nᵘ, length(param_nbs))
  mean_uₕ_test = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)

  for (i_nb, nb) in enumerate(param_nbs)
    println("\n")
    println("Considering parameter number: $nb/$(param_nbs[end])")

    Param = get_ParamInfo(RBInfo, FEMSpace, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:,(nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]

    mean_uₕ_test += uₕ_test

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
    mean_pointwise_err += abs.(uₕ_test-RBVars.ũ)/length(param_nbs)

    ũ_μ[:, (i_nb-1)*RBVars.Nₜ+1:i_nb*RBVars.Nₜ] = RBVars.ũ
    uₙ_μ[:, i_nb] = RBVars.uₙ

    println("Online wall time: $(RBVars.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)")
  end
  return (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,
    H1_L2_err,mean_online_time,mean_reconstruction_time)
end

function adaptive_loop_on_params(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::ADRUnsteady{T},
  mean_uₕ_test::Matrix,
  mean_pointwise_err::Matrix,
  μ::Vector{Vector{T}},
  param_nbs,
  n_adaptive=nothing) where T

  if isnothing(n_adaptive)
    nₛᵘ_add = floor(Int,RBVars.nₛᵘ*0.1)
    nₜᵘ_add = floor(Int,RBVars.nₜᵘ*0.1)
    n_adaptive = maximum(hcat([1,1],[nₛᵘ_add,nₜᵘ_add]),dims=2)::Vector{Int}
  end

  println("Running adaptive cycle: adding $n_adaptive temporal and spatial bases,
    respectively")

  time_err = zeros(T, RBVars.Nₜ)
  space_err = zeros(T, RBVars.Nₛᵘ)
  for iₜ = 1:RBVars.Nₜ
    time_err[iₜ] = (mynorm(mean_pointwise_err[:,iₜ],RBVars.Xᵘ₀) /
      mynorm(mean_uₕ_test[:,iₜ],RBVars.Xᵘ₀))
  end
  for iₛ = 1:RBVars.Nₛᵘ
    space_err[iₛ] = mynorm(mean_pointwise_err[iₛ,:])/mynorm(mean_uₕ_test[iₛ,:])
  end
  ind_s = argmax(space_err,n_adaptive[1])
  ind_t = argmax(time_err,n_adaptive[2])

  if isempty(RBVars.Sᵘ)
    Sᵘ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:,1:RBInfo.nₛ*RBVars.Nₜ]
  else
    Sᵘ = RBVars.Sᵘ
  end
  Sᵘ = reshape(sum(reshape(Sᵘ,RBVars.Nₛᵘ,RBVars.Nₜ,:),dims=3),RBVars.Nₛᵘ,:)

  Φₛᵘ_new = Matrix{T}(qr(Sᵘ[:,ind_t]).Q)[:,1:n_adaptive[2]]
  Φₜᵘ_new = Matrix{T}(qr(Sᵘ[ind_s,:]').Q)[:,1:n_adaptive[1]]
  RBVars.nₛᵘ += n_adaptive[2]
  RBVars.nₜᵘ += n_adaptive[1]
  RBVars.nᵘ = RBVars.nₛᵘ*RBVars.nₜᵘ

  RBVars.Φₛᵘ = Matrix{T}(qr(hcat(RBVars.Φₛᵘ,Φₛᵘ_new)).Q)[:,1:RBVars.nₛᵘ]
  RBVars.Φₜᵘ = Matrix{T}(qr(hcat(RBVars.Φₜᵘ,Φₜᵘ_new)).Q)[:,1:RBVars.nₜᵘ]
  RBInfo.save_offline_structures = false
  assemble_offline_structures(RBInfo, RBVars)

  loop_on_params(FEMSpace,RBInfo,RBVars,μ,param_nbs)

end
