include("RBPoisson_steady.jl")
include("ST-GRB_Poisson.jl")
include("ST-PGRB_Poisson.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T}) where T

  if RBInfo.perform_nested_POD
    println("Importing the snapshot matrix for field u obtained
      with the nested POD")
    Sᵘ = Matrix{T}(CSV.read(joinpath(RBInfo.paths.FEM_snap_path,"uₕ.csv"),
      DataFrame))
  else
    println("Importing the snapshot matrix for field u,
      number of snapshots considered: $(RBInfo.nₛ)")
    Sᵘ = Matrix{T}(CSV.read(joinpath(RBInfo.paths.FEM_snap_path,"uₕ.csv"),
      DataFrame))[:,1:RBInfo.nₛ*RBVars.Nₜ]
  end

  RBVars.S.Sᵘ = Sᵘ
  RBVars.S.Nₛᵘ = size(Sᵘ)[1]
  RBVars.Nᵘ = RBVars.S.Nₛᵘ * RBVars.Nₜ

  println("Dimension of the snapshot matrix for field u: $(size(Sᵘ))")

end

PODs_space(RBInfo::Info, RBVars::PoissonUnsteady) =
  PODs_space(RBInfo, RBVars.S)

function PODs_time(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T}) where T

  println("Performing the temporal POD for field u, using a tolerance of $(RBInfo.ϵₜ)")

  if RBInfo.time_reduction_technique == "ST-HOSVD"
    Sᵘₜ = zeros(T, RBVars.Nₜ, RBVars.S.nₛᵘ * RBInfo.nₛ)
    Sᵘ = RBVars.S.Φₛᵘ' * RBVars.S.Sᵘ
    @simd for i in 1:RBInfo.nₛ
      Sᵘₜ[:,(i-1)*RBVars.S.nₛᵘ+1:i*RBVars.S.nₛᵘ] =
      Sᵘ[:,(i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ]'
    end
  else
    Sᵘₜ = zeros(T, RBVars.Nₜ, RBVars.S.Nₛᵘ * RBInfo.nₛ)
    Sᵘ = RBVars.S.Sᵘ
    @simd for i in 1:RBInfo.nₛ
      Sᵘₜ[:, (i-1)*RBVars.S.Nₛᵘ+1:i*RBVars.S.Nₛᵘ] =
      transpose(Sᵘ[:, (i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ])
    end
  end

  Φₜᵘ, _ = POD(Sᵘₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵘ = Φₜᵘ
  RBVars.nₜᵘ = size(Φₜᵘ)[2]

end

function build_reduced_basis(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady)

  println("Building the space-time reduced basis for field u")

  RBVars.S.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
    PODs_time(RBInfo, RBVars)
  end

  RBVars.nᵘ = RBVars.S.nₛᵘ * RBVars.nₜᵘ
  RBVars.Nᵘ = RBVars.S.Nₛᵘ * RBVars.Nₜ

  if RBInfo.save_offline_structures
    save_CSV(RBVars.S.Φₛᵘ, joinpath(RBInfo.paths.basis_path, "Φₛᵘ.csv"))
    save_CSV(RBVars.Φₜᵘ, joinpath(RBInfo.paths.basis_path, "Φₜᵘ.csv"))
  end

end

function import_reduced_basis(
  RBInfo::Info,
  RBVars::PoissonUnsteady{T}) where T

  import_reduced_basis(RBInfo, RBVars.S)

  println("Importing the temporal reduced basis for field u")
  RBVars.Φₜᵘ = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.basis_path, "Φₜᵘ.csv"))
  RBVars.nₜᵘ = size(RBVars.Φₜᵘ)[2]
  RBVars.nᵘ = RBVars.S.nₛᵘ * RBVars.nₜᵘ

end

function index_mapping(
  i::Int64,
  j::Int64,
  RBVars::PoissonUnsteady)

  Int((i-1)*RBVars.nₜᵘ+j)

end

function get_generalized_coordinates(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  snaps::Vector{Int64}) where T

  if check_norm_matrix(RBVars.S)
    get_norm_matrix(RBInfo, RBVars.S)
  end

  @assert maximum(snaps) ≤ RBInfo.nₛ

  û = zeros(T, RBVars.nᵘ, length(snaps))
  Φₛᵘ_normed = RBVars.S.Xᵘ₀ * RBVars.S.Φₛᵘ
  Π = kron(Φₛᵘ_normed, RBVars.Φₜᵘ)::Matrix{T}

  for (i, i_nₛ) = enumerate(snaps)
    println("Assembling generalized coordinate relative to snapshot $(i_nₛ), field u")
    S_i = RBVars.S.Sᵘ[:, (i_nₛ-1)*RBVars.Nₜ+1:i_nₛ*RBVars.Nₜ]
    û[:, i] = sum(Π, dims=2) .* S_i
  end

  RBVars.S.û = û

  if RBInfo.save_offline_structures
    save_CSV(û, joinpath(RBInfo.paths.gen_coords_path, "û.csv"))
  end

end

function test_offline_phase(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T}) where T

  get_generalized_coordinates(RBInfo, RBVars, 1)

  uₙ = reshape(RBVars.S.û, (RBVars.nₜᵘ, RBVars.S.nₛᵘ))
  u_rec = RBVars.S.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'
  err = zeros(T, RBVars.Nₜ)
  for i = 1:RBVars.Nₜ
    err[i] = compute_errors(RBVars.S.Sᵘ[:, i], u_rec[:, i])
  end

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady,
  var::String)

  if var == "M"
    println("The matrix $var is non-affine:
      running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
    if isempty(RBVars.MDEIM_mat_M)
      (RBVars.MDEIM_mat_M, RBVars.MDEIM_idx_M, RBVars.MDEIMᵢ_M, RBVars.row_idx_M,
        RBVars.sparse_el_M) = MDEIM_offline(RBInfo, "M")
    end
    assemble_reduced_mat_MDEIM(
      RBInfo,RBVars,RBVars.MDEIM_mat_M,RBVars.row_idx_M,"M")
  elseif var == "A"
    if isempty(RBVars.S.MDEIM_mat_A)
      (RBVars.S.MDEIM_mat_A, RBVars.S.MDEIM_idx_A, RBVars.S.MDEIMᵢ_A,
      RBVars.S.row_idx_A,RBVars.S.sparse_el_A) = MDEIM_offline(RBInfo, "A")
    end
    assemble_reduced_mat_MDEIM(
      RBInfo,RBVars,RBVars.S.MDEIM_mat_A,RBVars.S.row_idx_A,"A")
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady,
  var::String)

  println("The vector $var is non-affine:
    running the DEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "F"
    if isempty(RBVars.S.DEIM_mat_F)
       RBVars.S.DEIM_mat_F, RBVars.S.DEIM_idx_F, RBVars.S.DEIMᵢ_F =
        DEIM_offline(RBInfo,"F")
    end
    assemble_reduced_mat_DEIM(RBInfo,RBVars,RBVars.S.DEIM_mat_F,"F")
  elseif var == "H"
    if isempty(RBVars.S.DEIM_mat_H)
       RBVars.S.DEIM_mat_H, RBVars.S.DEIM_idx_H, RBVars.S.DEIMᵢ_H =
        DEIM_offline(RBInfo,"H")
    end
    assemble_reduced_mat_DEIM(RBInfo,RBVars, RBVars.S.DEIM_mat_H,"H")
  else
    error("Unrecognized variable on which to perform DEIM")
  end

end

function save_M_DEIM_structures(
  RBInfo::Info,
  RBVars::PoissonUnsteady)

  list_M_DEIM = (RBVars.MDEIM_mat_M, RBVars.MDEIMᵢ_M, RBVars.MDEIM_idx_M,
    RBVars.sparse_el_M, RBVars.row_idx_M)
  list_names = ("MDEIM_mat_M", "MDEIMᵢ_M", "MDEIM_idx_M", "sparse_el_M",
   "row_idx_M")
  l_info_vec = [[l_idx,l_val] for (l_idx,l_val) in
    enumerate(list_M_DEIM) if !all(isempty.(l_val))]

  if !isempty(l_info_vec)
    l_info_mat = reduce(vcat,transpose.(l_info_vec))
    l_idx,l_val = l_info_mat[:,1], transpose.(l_info_mat[:,2])
    for (i₁,i₂) in enumerate(l_idx)
      save_CSV(l_val[i₁], joinpath(RBInfo.paths.ROM_structures_path,
        list_names[i₂]*".csv"))
    end
  end

  save_M_DEIM_structures(RBInfo, RBVars.S)

end

function set_operators(
  RBInfo::Info,
  RBVars::PoissonUnsteady)

  vcat(["M"], set_operators(RBInfo, RBVars.S))

end

function get_M_DEIM_structures(
  RBInfo::Info,
  RBVars::PoissonUnsteady{T}) where T

  operators = String[]

  if RBInfo.probl_nl["M"]

    if isfile(joinpath(RBInfo.paths.ROM_structures_path, "MDEIMᵢ_M.csv"))
      println("Importing MDEIM offline structures for the mass matrix")
      RBVars.MDEIMᵢ_M = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.ROM_structures_path,
        "MDEIMᵢ_M.csv"))
      RBVars.MDEIM_idx_M = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.ROM_structures_path,
        "MDEIM_idx_M.csv"))[:]
      RBVars.sparse_el_M = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.ROM_structures_path,
        "sparse_el_M.csv"))[:]
      RBVars.row_idx_M = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.paths.ROM_structures_path,
        "row_idx_M.csv"))[:]
      append!(operators, [])
    else
      println("Failed to import MDEIM offline structures for the mass matrix: must build them")
      append!(operators, ["M"])
    end

  end

  append!(operators, get_M_DEIM_structures(RBInfo, RBVars.S))

end

function get_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady)

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_θᵐ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady) where T

  timesθ = get_timesθ(RBInfo)

  if !RBInfo.probl_nl["M"]
    θᵐ = [Param.mₜ(t_θ) for t_θ = timesθ]
  else
    M_μ_sparse = build_sparse_mat(
      FEMSpace,FEMInfo,Param,RBVars.sparse_el_M,timesθ;var="M")
    θᵐ = (RBVars.MDEIMᵢ_M \
      Matrix{T}(reshape(M_μ_sparse, :, RBVars.Nₜ)[RBVars.MDEIM_idx_M, :]))
  end

  reshape(θᵐ, RBVars.Qᵐ, RBVars.Nₜ)::Matrix{T}

end

function get_θᵐₛₜ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady) where T

  if !RBInfo.probl_nl["M"]
    θᵐ = [one(T)]
  else
    timesθ_mod,MDEIM_idx_mod =
      modify_timesθ_and_MDEIM_idx(RBVars.MDEIM_idx_M,RBInfo,RBVars)
      M_μ_sparse = build_sparse_mat(FEMSpace, FEMInfo, Param, RBVars.sparse_el_M,
      timesθ_mod; var="M")
    θᵐ = RBVars.MDEIMᵢ_M\Vector(M_μ_sparse[MDEIM_idx_mod])
  end

  reshape(θᵐ,:,1)::Matrix{T}

end

function get_θᵃ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady) where T

  timesθ = get_timesθ(RBInfo)

  if !RBInfo.probl_nl["A"]
    θᵃ = [Param.αₜ(t_θ,Param.μ) for t_θ = timesθ]
  else
    A_μ_sparse = build_sparse_mat(
      FEMSpace,FEMInfo,Param,RBVars.S.sparse_el_A,timesθ;var="A")
    θᵃ = (RBVars.S.MDEIMᵢ_A \
      Matrix{T}(reshape(A_μ_sparse, :, RBVars.Nₜ)[RBVars.S.MDEIM_idx_A, :]))
  end

  reshape(θᵃ, RBVars.S.Qᵃ, RBVars.Nₜ)::Matrix{T}

end

function get_θᵃₛₜ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady) where T

  if !RBInfo.probl_nl["A"]
    θᵃ = [one(T)]
  else
    timesθ_mod,MDEIM_idx_mod =
      modify_timesθ_and_MDEIM_idx(RBVars.S.MDEIM_idx_A,RBInfo,RBVars)
    A_μ_sparse = build_sparse_mat(FEMSpace,FEMInfo, Param, RBVars.S.sparse_el_A,
      timesθ_mod; var="A")
    θᵃ = RBVars.S.MDEIMᵢ_A\Vector(A_μ_sparse[MDEIM_idx_mod])
  end

  reshape(θᵃ, :, 1)::Matrix{T}

end

function get_θᶠʰ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady) where T

  if RBInfo.build_parametric_RHS
    error("Cannot fetch θᶠ, θʰ if the RHS is built online")
  end

  timesθ = get_timesθ(RBInfo)

  if !RBInfo.probl_nl["f"]
    θᶠ = [Param.fₜ(t_θ) for t_θ = timesθ]
  else
    F_μ = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
    F = zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ)
    for (i,tᵢ) in enumerate(timesθ)
      F[:,i] = F_μ(tᵢ)
    end
    θᶠ = (RBVars.S.DEIMᵢ_F \ Matrix{T}(F[RBVars.S.DEIM_idx_F, :]))
  end

  if !RBInfo.probl_nl["h"]
    θʰ = [Param.hₜ(t_θ) for t_θ = timesθ]
  else
    H_μ = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
    H = zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ)
    for (i,tᵢ) in enumerate(timesθ)
      H[:,i] = H_μ(tᵢ)
    end
    θʰ = (RBVars.S.DEIMᵢ_H \ Matrix{T}(H[RBVars.S.DEIM_idx_H, :]))
  end

  (reshape(θᶠ, RBVars.S.Qᶠ, RBVars.Nₜ)::Matrix{T},
  reshape(θʰ, RBVars.S.Qʰ, RBVars.Nₜ)::Matrix{T})

end

function get_θᶠʰₛₜ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady) where T

  if RBInfo.build_parametric_RHS
    error("Cannot fetch θᶠ, θʰ if the RHS is built online")
  end

  if !RBInfo.probl_nl["f"]
    θᶠ = [one(T)]
  else
    F_μ = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
    F = zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ)
    for (i,tᵢ) in enumerate(timesθ)
      F[:,i] = F_μ(tᵢ)
    end
    _,DEIM_idx_mod = modify_timesθ_and_MDEIM_idx(RBVars.S.DEIM_idx_F,RBInfo,RBVars)
    θᶠ = T.(RBVars.S.DEIMᵢ_F\Vector(F[DEIM_idx_mod]))
  end

  if !RBInfo.probl_nl["h"]
    θʰ = [one(T)]
  else
    H_μ = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
    H = zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ)
    for (i,tᵢ) in enumerate(timesθ)
      H[:,i] = H_μ(tᵢ)
    end
    _,DEIM_idx_mod = modify_timesθ_and_MDEIM_idx(RBVars.S.DEIM_idx_H,RBInfo,RBVars)
    θʰ = T.(RBVars.S.DEIMᵢ_H\Vector(H[DEIM_idx_mod]))
  end

  reshape(θᶠ, :, 1)::Matrix{T}, reshape(θʰ, :, 1)::Matrix{T}

end

function solve_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady,
  Param::ParametricInfoUnsteady)

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.S.LHSₙ[1]))")

  RBVars.S.online_time += @elapsed begin
    @fastmath RBVars.S.uₙ = RBVars.S.LHSₙ[1] \ RBVars.S.RHSₙ[1]
  end

end

function reconstruct_FEM_solution(RBVars::PoissonUnsteady)

  println("Reconstructing FEM solution from the newly computed RB one")
  uₙ = reshape(RBVars.S.uₙ, (RBVars.nₜᵘ, RBVars.S.nₛᵘ))
  @fastmath RBVars.S.ũ = RBVars.S.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'

end

function offline_phase(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady)

  println("Offline phase of the RB solver, unsteady Poisson problem")

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
  RBVars::PoissonUnsteady,
  param_nbs) where T

  println("Online phase of the RB solver, unsteady Poisson problem")

  μ = load_CSV(Array{T}[],
    joinpath(RBInfo.paths.FEM_snap_path, "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(RBInfo.paths.mesh_path)
  FEMSpace = get_FEMSpace₀(RBInfo.FEMInfo.problem_id,RBInfo.FEMInfo,model)

  get_norm_matrix(RBInfo, RBVars.S)
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
  path_μ = joinpath(RBInfo.paths.results_path, string_param_nbs)

  if RBInfo.save_results
    println("Saving the results...")
    create_dir(path_μ)
    save_CSV(ũ_μ, joinpath(path_μ, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(path_μ, "uₙ.csv"))
    save_CSV(mean_pointwise_err, joinpath(path_μ, "mean_point_err.csv"))
    save_CSV(mean_H1_err, joinpath(path_μ, "H1_err.csv"))
    save_CSV([mean_H1_L2_err], joinpath(path_μ, "H1L2_err.csv"))

    times = Dict("off_time"=>RBVars.S.offline_time,
      "on_time"=>mean_online_time+adapt_time,"rec_time"=>mean_reconstruction_time)
    CSV.write(joinpath(path_μ, "times.csv"),times)
  end

  pass_to_pp = Dict("path_μ"=>path_μ,
    "FEMSpace"=>FEMSpace, "H1_L2_err"=>H1_L2_err,
    "mean_H1_err"=>mean_H1_err, "mean_point_err_u"=>mean_pointwise_err)

  if RBInfo.post_process
    println("Post-processing the results...")
    post_process(RBInfo, pass_to_pp)
  end

  #=
  plot_stability_constant(FEMSpace,RBInfo,Param,Nₜ)
  =#

end

function loop_on_params(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  μ::Vector{Vector{T}},
  param_nbs) where T

  H1_L2_err = zeros(T, length(param_nbs))
  mean_H1_err = zeros(T, RBVars.Nₜ)
  mean_H1_L2_err = 0.0
  mean_pointwise_err = zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  ũ_μ = zeros(T, RBVars.S.Nₛᵘ, length(param_nbs)*RBVars.Nₜ)
  uₙ_μ = zeros(T, RBVars.nᵘ, length(param_nbs))
  mean_uₕ_test = zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ)

  for (i_nb, nb) in enumerate(param_nbs)
    println("\n")
    println("Considering Parameter number: $nb/$(param_nbs[end])")

    Param = get_ParamInfo(RBInfo, μ[nb])
    if RBInfo.perform_nested_POD
      nb_test = nb-90
      uₕ_test = Matrix{T}(CSV.read(joinpath(RBInfo.paths.FEM_snap_path,
      "uₕ_test.csv"), DataFrame))[:,(nb_test-1)*RBVars.Nₜ+1:nb_test*RBVars.Nₜ]
    else
      uₕ_test = Matrix{T}(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"),
      DataFrame))[:,(nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]
    end
    mean_uₕ_test += uₕ_test

    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    if i_nb > 1
      mean_online_time = RBVars.S.online_time/(length(param_nbs)-1)
      mean_reconstruction_time = reconstruction_time/(length(param_nbs)-1)
    end

    H1_err_nb, H1_L2_err_nb = compute_errors(uₕ_test, RBVars, RBVars.S.Xᵘ₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err += abs.(uₕ_test-RBVars.S.ũ)/length(param_nbs)

    ũ_μ[:, (i_nb-1)*RBVars.Nₜ+1:i_nb*RBVars.Nₜ] = RBVars.S.ũ
    uₙ_μ[:, i_nb] = RBVars.S.uₙ

    println("Online wall time: $(RBVars.S.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-L2 error: $H1_L2_err_nb (snapshot number $nb)")
  end
  return (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,
    H1_L2_err,mean_online_time,mean_reconstruction_time)
end

function adaptive_loop_on_params(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  mean_uₕ_test::Matrix,
  mean_pointwise_err::Matrix,
  μ::Vector{Vector{T}},
  param_nbs,
  n_adaptive=nothing) where T

  if isnothing(n_adaptive)
    nₛᵘ_add = floor(Int64,RBVars.S.nₛᵘ*0.1)
    nₜᵘ_add = floor(Int64,RBVars.nₜᵘ*0.1)
    n_adaptive = maximum(hcat([1,1],[nₛᵘ_add,nₜᵘ_add]),dims=2)::Vector{Int}
  end

  println("Running adaptive cycle: adding $n_adaptive temporal and spatial bases,
    respectively")

  time_err = zeros(T, RBVars.Nₜ)
  space_err = zeros(T, RBVars.S.Nₛᵘ)
  for iₜ = 1:RBVars.Nₜ
    time_err[iₜ] = (mynorm(mean_pointwise_err[:,iₜ],RBVars.S.Xᵘ₀) /
      mynorm(mean_uₕ_test[:,iₜ],RBVars.S.Xᵘ₀))
  end
  for iₛ = 1:RBVars.S.Nₛᵘ
    space_err[iₛ] = mynorm(mean_pointwise_err[iₛ,:])/mynorm(mean_uₕ_test[iₛ,:])
  end
  ind_s = argmax(space_err,n_adaptive[1])
  ind_t = argmax(time_err,n_adaptive[2])

  if isempty(RBVars.S.Sᵘ)
    Sᵘ = Matrix{T}(CSV.read(joinpath(RBInfo.paths.FEM_snap_path, "uₕ.csv"),
      DataFrame))[:,1:RBInfo.nₛ*RBVars.Nₜ]
  else
    Sᵘ = RBVars.S.Sᵘ
  end
  Sᵘ = reshape(sum(reshape(Sᵘ,RBVars.S.Nₛᵘ,RBVars.Nₜ,:),dims=3),RBVars.S.Nₛᵘ,:)

  Φₛᵘ_new = Matrix{T}(qr(Sᵘ[:,ind_t]).Q)[:,1:n_adaptive[2]]
  Φₜᵘ_new = Matrix{T}(qr(Sᵘ[ind_s,:]').Q)[:,1:n_adaptive[1]]
  RBVars.S.nₛᵘ += n_adaptive[2]
  RBVars.nₜᵘ += n_adaptive[1]
  RBVars.nᵘ = RBVars.S.nₛᵘ*RBVars.nₜᵘ

  RBVars.S.Φₛᵘ = Matrix{T}(qr(hcat(RBVars.S.Φₛᵘ,Φₛᵘ_new)).Q)[:,1:RBVars.S.nₛᵘ]
  RBVars.Φₜᵘ = Matrix{T}(qr(hcat(RBVars.Φₜᵘ,Φₜᵘ_new)).Q)[:,1:RBVars.nₜᵘ]
  RBInfo.save_offline_structures = false
  assemble_offline_structures(RBInfo, RBVars)

  loop_on_params(FEMSpace,RBInfo,RBVars,μ,param_nbs)

end
