include("RBPoisson_steady.jl")
include("ST-GRB_Poisson.jl")
include("ST-PGRB_Poisson.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T}) where T

  println("Importing the snapshot matrix for field u,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵘ = Matrix{T}(CSV.read(joinpath(RBInfo.Paths.FEM_snap_path,"uₕ.csv"),
    DataFrame))[:,1:RBInfo.nₛ*RBVars.Nₜ]

  RBVars.Sᵘ = Sᵘ
  RBVars.Nₛᵘ = size(Sᵘ)[1]
  RBVars.Nᵘ = RBVars.Nₛᵘ * RBVars.Nₜ

  println("Dimension of the snapshot matrix for field u: $(size(Sᵘ))")

end

PODs_space(RBInfo::Info, RBVars::PoissonUnsteady) =
  PODs_space(RBInfo, RBVars.S)

function PODs_time(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T}) where T

  println("Performing the temporal POD for field u, using a tolerance of $(RBInfo.ϵₜ)")

  if RBInfo.time_reduction_technique == "ST-HOSVD"
    Sᵘₜ = zeros(T, RBVars.Nₜ, RBVars.nₛᵘ * RBInfo.nₛ)
    Sᵘ = RBVars.Φₛᵘ' * RBVars.Sᵘ
    @simd for i in 1:RBInfo.nₛ
      Sᵘₜ[:,(i-1)*RBVars.nₛᵘ+1:i*RBVars.nₛᵘ] =
      Sᵘ[:,(i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ]'
    end
  else
    Sᵘₜ = zeros(T, RBVars.Nₜ, RBVars.Nₛᵘ * RBInfo.nₛ)
    Sᵘ = RBVars.Sᵘ
    @simd for i in 1:RBInfo.nₛ
      Sᵘₜ[:, (i-1)*RBVars.Nₛᵘ+1:i*RBVars.Nₛᵘ] =
      transpose(Sᵘ[:, (i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ])
    end
  end

  Φₜᵘ, _ = POD(Sᵘₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵘ = Φₜᵘ
  RBVars.nₜᵘ = size(Φₜᵘ)[2]

end

#= function PODs_time_old(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T}) where T

  println("Performing the temporal POD for field u, using a tolerance of $(RBInfo.ϵₜ)")

  if RBInfo.time_reduction_technique == "ST-HOSVD"
    Sᵘₜ = zeros(T, RBVars.Nₜ, RBVars.S.nₛᵘ * RBInfo.S.nₛ)
    Sᵘ = RBVars.S.Φₛᵘ' * RBVars.S.Sᵘ
    @simd for i in 1:RBInfo.S.nₛ
      Sᵘₜ[:,(i-1)*RBVars.S.nₛᵘ+1:i*RBVars.S.nₛᵘ] =
      Sᵘ[:,(i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ]'
    end
  else
    Sᵘₜ = zeros(T, RBVars.Nₜ, RBVars.S.Nₛᵘ * RBInfo.S.nₛ)
    Sᵘ = RBVars.S.Sᵘ
    @simd for i in 1:RBInfo.S.nₛ
      Sᵘₜ[:, (i-1)*RBVars.Nₛᵘ+1:i*RBVars.S.Nₛᵘ] =
      transpose(Sᵘ[:, (i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ])
    end
  end

  Φₜᵘ, _ = POD(Sᵘₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵘ = Φₜᵘ
  RBVars.nₜᵘ = size(Φₜᵘ)[2]

end =#

function build_reduced_basis(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady)

  println("Building the space-time reduced basis for field u")

  RBVars.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
    PODs_time(RBInfo, RBVars)
  end

  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ
  RBVars.Nᵘ = RBVars.Nₛᵘ * RBVars.Nₜ

  if RBInfo.save_offline_structures
    save_CSV(RBVars.Φₛᵘ, joinpath(RBInfo.Paths.basis_path, "Φₛᵘ.csv"))
    save_CSV(RBVars.Φₜᵘ, joinpath(RBInfo.Paths.basis_path, "Φₜᵘ.csv"))
  end

  return

end

function import_reduced_basis(
  RBInfo::Info,
  RBVars::PoissonUnsteady{T}) where T

  import_reduced_basis(RBInfo, RBVars.S)

  println("Importing the temporal reduced basis for field u")
  RBVars.Φₜᵘ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.basis_path, "Φₜᵘ.csv"))
  RBVars.nₜᵘ = size(RBVars.Φₜᵘ)[2]
  RBVars.nᵘ = RBVars.nₛᵘ * RBVars.nₜᵘ

end

function index_mapping(
  i::Int,
  j::Int,
  RBVars::PoissonUnsteady)

  Int((i-1)*RBVars.nₜᵘ+j)

end

function get_generalized_coordinates(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  snaps::Vector{Int}) where T

  if check_norm_matrix(RBVars.S)
    get_norm_matrix(RBInfo, RBVars.S)
  end

  @assert maximum(snaps) ≤ RBInfo.nₛ

  û = zeros(T, RBVars.nᵘ, length(snaps))
  Φₛᵘ_normed = RBVars.Xᵘ₀ * RBVars.Φₛᵘ
  Π = kron(Φₛᵘ_normed, RBVars.Φₜᵘ)::Matrix{T}

  for (i, i_nₛ) = enumerate(snaps)
    println("Assembling generalized coordinate relative to snapshot $(i_nₛ), field u")
    S_i = RBVars.Sᵘ[:, (i_nₛ-1)*RBVars.Nₜ+1:i_nₛ*RBVars.Nₜ]
    û[:, i] = sum(Π, dims=2) .* S_i
  end

  RBVars.û = û

  if RBInfo.save_offline_structures
    save_CSV(û, joinpath(RBInfo.Paths.gen_coords_path, "û.csv"))
  end

end

function test_offline_phase(
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T}) where T

  get_generalized_coordinates(RBInfo, RBVars, 1)

  uₙ = reshape(RBVars.û, (RBVars.nₜᵘ, RBVars.nₛᵘ))
  u_rec = RBVars.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'
  err = zeros(T, RBVars.Nₜ)
  for i = 1:RBVars.Nₜ
    err[i] = compute_errors(RBVars.Sᵘ[:, i], u_rec[:, i])
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
        RBVars.sparse_el_M, RBVars.MDEIM_idx_time_M) = MDEIM_offline(RBInfo, "M")
    end
    assemble_reduced_mat_MDEIM(
      RBVars,RBVars.MDEIM_mat_M,RBVars.row_idx_M,"M")
  elseif var == "A"
    if isempty(RBVars.MDEIM_mat_A)
      (RBVars.MDEIM_mat_A, RBVars.MDEIM_idx_A, RBVars.MDEIMᵢ_A,
      RBVars.row_idx_A,RBVars.sparse_el_A, RBVars.MDEIM_idx_time_A) = MDEIM_offline(RBInfo, "A")
    end
    assemble_reduced_mat_MDEIM(
      RBVars,RBVars.MDEIM_mat_A,RBVars.row_idx_A,"A")
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
    if isempty(RBVars.DEIM_mat_F)
       (RBVars.DEIM_mat_F, RBVars.DEIM_idx_F, RBVars.DEIMᵢ_F,
          RBVars.sparse_el_F, RBVars.DEIM_idx_time_F) = DEIM_offline(RBInfo,"F")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_F,"F")
  elseif var == "H"
    if isempty(RBVars.DEIM_mat_H)
       (RBVars.DEIM_mat_H, RBVars.DEIM_idx_H, RBVars.DEIMᵢ_H,
          RBVars.sparse_el_H, RBVars.DEIM_idx_time_H) = DEIM_offline(RBInfo,"H")
    end
    assemble_reduced_mat_DEIM(RBVars, RBVars.DEIM_mat_H,"H")
  else
    error("Unrecognized variable on which to perform DEIM")
  end

end

function save_M_DEIM_structures(
  RBInfo::Info,
  RBVars::PoissonUnsteady)

  list_M_DEIM = (RBVars.MDEIM_mat_M, RBVars.MDEIMᵢ_M, RBVars.MDEIM_idx_M,
    RBVars.sparse_el_M, RBVars.row_idx_M, RBVars.MDEIM_idx_time_A,
    RBVars.MDEIM_idx_time_M, RBVars.DEIM_idx_time_F, RBVars.DEIM_idx_time_H)
  list_names = ("MDEIM_mat_M", "MDEIMᵢ_M", "MDEIM_idx_M", "sparse_el_M",
   "row_idx_M", "MDEIM_idx_time_A", "MDEIM_idx_time_M", "DEIM_idx_time_F", "DEIM_idx_time_H")
  l_info_vec = [[l_idx,l_val] for (l_idx,l_val) in
    enumerate(list_M_DEIM) if !all(isempty.(l_val))]

  if !isempty(l_info_vec)
    l_info_mat = reduce(vcat,transpose.(l_info_vec))
    l_idx,l_val = l_info_mat[:,1], transpose.(l_info_mat[:,2])
    for (i₁,i₂) in enumerate(l_idx)
      save_CSV(l_val[i₁], joinpath(RBInfo.Paths.ROM_structures_path,
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

  if RBInfo.probl_nl["A"]
    if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "MDEIM_idx_time_A.csv"))
      RBVars.MDEIM_idx_time_A = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.Paths.ROM_structures_path, "MDEIM_idx_time_A.csv"))
    else
      append!(operators, ["A"])
    end
  end

  if RBInfo.probl_nl["M"]

    if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "MDEIMᵢ_M.csv"))
      println("Importing MDEIM offline structures for the mass matrix")
      RBVars.MDEIMᵢ_M = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "MDEIMᵢ_M.csv"))
      RBVars.MDEIM_idx_M = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "MDEIM_idx_M.csv"))
      RBVars.sparse_el_M = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "sparse_el_M.csv"))
      RBVars.row_idx_M = load_CSV(Vector{Int}(undef,0), joinpath(RBInfo.Paths.ROM_structures_path,
        "row_idx_M.csv"))
      RBVars.MDEIM_idx_time_M = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.Paths.ROM_structures_path, "MDEIM_idx_time_M.csv"))
      append!(operators, [])
    else
      println("Failed to import MDEIM offline structures for the mass matrix: must build them")
      append!(operators, ["M"])
    end

  end

  if RBInfo.probl_nl["f"]
    if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "DEIM_idx_time_F.csv"))
    RBVars.DEIM_idx_time_F = load_CSV(Vector{Int}(undef,0),
      joinpath(RBInfo.Paths.ROM_structures_path, "DEIM_idx_time_F.csv"))
    else
      append!(operators, ["F"])
    end
  end

  if RBInfo.probl_nl["h"]
    if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "DEIM_idx_time_H.csv"))
    RBVars.DEIM_idx_time_H = load_CSV(Vector{Int}(undef,0),
      joinpath(RBInfo.Paths.ROM_structures_path, "DEIM_idx_time_H.csv"))
    else
      append!(operators, ["H"])
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

function interpolated_θ(
  RBVars::PoissonUnsteady{T},
  Mat_μ_sparse::SparseMatrixCSC{T, Int},
  timesθ::Vector{T},
  MDEIMᵢ::Matrix{T},
  MDEIM_idx::Vector{Int},
  MDEIM_idx_time::Vector{Int},
  Q::Int) where T

  red_timesθ = timesθ[MDEIM_idx_time]
  discarded_idx_time = setdiff(collect(1:RBVars.Nₜ), MDEIM_idx_time)
  θ = zeros(T, Q, RBVars.Nₜ)

  red_θ = (MDEIMᵢ \
    Matrix{T}(reshape(Mat_μ_sparse, :, length(red_timesθ))[MDEIM_idx, :]))

  etp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ,1,:), red_θ')
  θ[:, MDEIM_idx_time] = red_θ
  for iₜ = discarded_idx_time
    θ[:, iₜ] = ScatteredInterpolation.evaluate(etp,[timesθ[iₜ]])
  end

  θ::Matrix{T}

end

function interpolated_θ(
  RBVars::PoissonUnsteady{T},
  Vec_μ_sparse::Matrix{T},
  timesθ::Vector{T},
  DEIMᵢ::Matrix{T},
  DEIM_idx::Vector{Int},
  DEIM_idx_time::Vector{Int},
  Q::Int) where T

  red_timesθ = timesθ[DEIM_idx_time]
  discarded_idx_time = setdiff(collect(1:RBVars.Nₜ), DEIM_idx_time)
  θ = zeros(T, Q, RBVars.Nₜ)

  red_θ = (DEIMᵢ \
    Matrix{T}(reshape(Vec_μ_sparse, :, length(red_timesθ))[DEIM_idx, :]))

  etp = ScatteredInterpolation.interpolate(Multiquadratic(),
    reshape(red_timesθ,1,:), red_θ')
  θ[:, DEIM_idx_time] = red_θ
  for iₜ = discarded_idx_time
    θ[:, iₜ] = ScatteredInterpolation.evaluate(etp,[timesθ[iₜ]])
  end

  θ::Matrix{T}

end

function get_θᵐ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady) where T

  timesθ = get_timesθ(RBInfo)

  if !RBInfo.probl_nl["M"]
    θᵐ = T.(zeros(T, 1, RBVars.Nₜ))
    for (i_t, t_θ) = enumerate(timesθ)
      θᵐ[i_t] = Param.mₜ(t_θ)
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[RBVars.MDEIM_idx_time_M]
      M_μ_sparse = T.(build_sparse_mat(
        FEMSpace,FEMInfo,Param,RBVars.sparse_el_M,red_timesθ;var="M"))
      θᵐ = interpolated_θ(RBVars, M_μ_sparse, timesθ, RBVars.MDEIMᵢ_M,
        RBVars.MDEIM_idx_M, RBVars.MDEIM_idx_time_M, RBVars.Qᵐ)
    else
      M_μ_sparse = T.(build_sparse_mat(
        FEMSpace,FEMInfo,Param,RBVars.sparse_el_M,timesθ;var="M"))
      θᵐ = (RBVars.MDEIMᵢ_M \
        Matrix{T}(reshape(M_μ_sparse, :, RBVars.Nₜ)[RBVars.MDEIM_idx_M, :]))
    end
  end

  θᵐ::Matrix{T}

end

function get_θᵃ(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady) where T

  timesθ = get_timesθ(RBInfo)

  if !RBInfo.probl_nl["A"]
    θᵃ = zeros(T, 1, RBVars.Nₜ)
    for (i_t, t_θ) = enumerate(timesθ)
      θᵃ[i_t] = T.(Param.αₜ(t_θ,Param.μ))
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[RBVars.MDEIM_idx_time_A]
      A_μ_sparse = T.(build_sparse_mat(
        FEMSpace,FEMInfo,Param,RBVars.sparse_el_A,red_timesθ;var="A"))
      θᵃ = interpolated_θ(RBVars, A_μ_sparse, timesθ, RBVars.MDEIMᵢ_A,
        RBVars.MDEIM_idx_A, RBVars.MDEIM_idx_time_A, RBVars.Qᵃ)
    else
      A_μ_sparse = build_sparse_mat(
        FEMSpace,FEMInfo,Param,RBVars.sparse_el_A,timesθ;var="A")
      θᵃ = (RBVars.MDEIMᵢ_A \
        Matrix{T}(reshape(A_μ_sparse, :, RBVars.Nₜ)[RBVars.MDEIM_idx_A, :]))
    end
  end

  save_CSV(θᵃ, joinpath(RBInfo.Paths.ROM_structures_path, "θᵃ.csv"))
  θᵃ::Matrix{T}

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
    θᶠ = zeros(T, 1, RBVars.Nₜ)
    for (i_t, t_θ) = enumerate(timesθ)
      θᶠ[i_t] = T.(Param.fₜ(t_θ))
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[RBVars.DEIM_idx_time_F]
      F_μ_sparse = T.(build_sparse_vec(
        FEMSpace,FEMInfo, Param, RBVars.sparse_el_F, red_timesθ; var="F"))
      θᶠ = interpolated_θ(RBVars, F_μ_sparse, timesθ, RBVars.DEIMᵢ_F,
        RBVars.DEIM_idx_F, RBVars.DEIM_idx_time_F, RBVars.Qᶠ)
    else
      F_μ = build_sparse_vec(FEMSpace,FEMInfo, Param, RBVars.sparse_el_F,
        timesθ; var="F")
      θᶠ = (RBVars.DEIMᵢ_F \ Matrix{T}(F_μ[RBVars.DEIM_idx_F, :]))
    end
  end

  if !RBInfo.probl_nl["h"]
    θʰ = zeros(T, 1, RBVars.Nₜ)
    for (i_t, t_θ) = enumerate(timesθ)
      θʰ[i_t] = T.(Param.hₜ(t_θ))
    end
  else
    if RBInfo.st_M_DEIM
      red_timesθ = timesθ[RBVars.DEIM_idx_time_H]
      H_μ_sparse = T.(build_sparse_vec(
        FEMSpace,FEMInfo, Param, RBVars.sparse_el_H, red_timesθ; var="H"))
      θʰ =  interpolated_θ(RBVars, H_μ_sparse, timesθ, RBVars.DEIMᵢ_H,
        RBVars.DEIM_idx_H, RBVars.DEIM_idx_time_H, RBVars.Qʰ)
    else
      H_μ = build_sparse_vec(FEMSpace,FEMInfo, Param, RBVars.sparse_el_H,
        timesθ; var="H")
      θʰ = (RBVars.DEIMᵢ_H \ Matrix{T}(H_μ[RBVars.DEIM_idx_H, :]))
    end
  end

  (θᶠ, θʰ)::Tuple{Matrix{T},Matrix{T}}

end

function solve_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::PoissonUnsteady,
  Param::ParametricInfoUnsteady)

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.LHSₙ[1]))")

  RBVars.online_time += @elapsed begin
    @fastmath RBVars.uₙ = RBVars.LHSₙ[1] \ RBVars.RHSₙ[1]
  end

end

function reconstruct_FEM_solution(RBVars::PoissonUnsteady)

  println("Reconstructing FEM solution from the newly computed RB one")
  uₙ = reshape(RBVars.uₙ, (RBVars.nₜᵘ, RBVars.nₛᵘ))
  @fastmath RBVars.ũ = RBVars.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'

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
    joinpath(RBInfo.Paths.FEM_snap_path, "μ.csv"))::Vector{Vector{T}}
  model = DiscreteModelFromFile(RBInfo.Paths.mesh_path)
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
  path_μ = joinpath(RBInfo.Paths.results_path, string_param_nbs)

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
  RBVars::PoissonUnsteady{T},
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
    println("Considering Parameter number: $nb/$(param_nbs[end])")

    Param = get_ParamInfo(RBInfo, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(RBInfo.Paths.FEM_snap_path, "uₕ.csv"),
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
  RBVars::PoissonUnsteady{T},
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
    Sᵘ = Matrix{T}(CSV.read(joinpath(RBInfo.Paths.FEM_snap_path, "uₕ.csv"),
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
