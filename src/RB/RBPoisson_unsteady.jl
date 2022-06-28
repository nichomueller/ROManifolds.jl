include("RBPoisson_steady.jl")
include("ST-GRB_Poisson.jl")
include("ST-PGRB_Poisson.jl")

function get_snapshot_matrix(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T}) where T

  if RBInfo.perform_nested_POD
    println("Importing the snapshot matrix for field u obtained
      with the nested POD")
    Sᵘ = Matrix{T}(CSV.read(joinpath(RBInfo.S.paths.FEM_snap_path,"uₕ.csv"),
      DataFrame))
  else
    println("Importing the snapshot matrix for field u,
      number of snapshots considered: $(RBInfo.S.nₛ)")
    Sᵘ = Matrix{T}(CSV.read(joinpath(RBInfo.S.paths.FEM_snap_path,"uₕ.csv"),
      DataFrame))[:,1:RBInfo.S.nₛ*RBVars.Nₜ]
  end

  RBVars.S.Sᵘ = Sᵘ
  Nₛᵘ = size(Sᵘ)[1]
  RBVars.S.Nₛᵘ = Nₛᵘ
  RBVars.Nᵘ = RBVars.S.Nₛᵘ * RBVars.Nₜ

  println("Dimension of the snapshot matrix for field u: $(size(Sᵘ))")

end

PODs_space(RBInfo::ROMInfoUnsteady{T}, RBVars::PoissonUnsteady{T}) where T =
  PODs_space(RBInfo.S, RBVars.S)

function PODs_time(
  RBInfo::ROMInfoUnsteady{T},
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
      Sᵘₜ[:, (i-1)*RBVars.S.Nₛᵘ+1:i*RBVars.S.Nₛᵘ] =
      transpose(Sᵘ[:, (i-1)*RBVars.Nₜ+1:i*RBVars.Nₜ])
    end
  end

  Φₜᵘ, _ = POD(Sᵘₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵘ = Φₜᵘ
  RBVars.nₜᵘ = size(Φₜᵘ)[2]

end

function build_reduced_basis(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T}) where T

  println("Building the space-time reduced basis for field u")

  RBVars.S.offline_time += @elapsed begin
    PODs_space(RBInfo, RBVars)
    PODs_time(RBInfo, RBVars)
  end

  RBVars.nᵘ = RBVars.S.nₛᵘ * RBVars.nₜᵘ
  RBVars.Nᵘ = RBVars.S.Nₛᵘ * RBVars.Nₜ

  if RBInfo.S.save_offline_structures
    save_CSV(RBVars.S.Φₛᵘ, joinpath(RBInfo.S.paths.basis_path, "Φₛᵘ.csv"))
    save_CSV(RBVars.Φₜᵘ, joinpath(RBInfo.S.paths.basis_path, "Φₜᵘ.csv"))
  end

end

function import_reduced_basis(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T}) where T

  import_reduced_basis(RBInfo.S, RBVars.S)

  println("Importing the temporal reduced basis for field u")
  RBVars.Φₜᵘ = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.S.paths.basis_path, "Φₜᵘ.csv"))
  RBVars.nₜᵘ = size(RBVars.Φₜᵘ)[2]
  RBVars.nᵘ = RBVars.S.nₛᵘ * RBVars.nₜᵘ

end

function index_mapping(
  i::Int64,
  j::Int64,
  RBVars::PoissonUnsteady{T}) where T

  Int((i-1)*RBVars.nₜᵘ+j)

end

function get_generalized_coordinates(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  snaps::Vector{Int64}) where T

  if check_norm_matrix(RBVars.S)
    get_norm_matrix(RBInfo.S, RBVars.S)
  end

  @assert maximum(snaps) ≤ RBInfo.S.nₛ

  û = zeros(T, RBVars.nᵘ, length(snaps))
  Φₛᵘ_normed = RBVars.S.Xᵘ₀ * RBVars.S.Φₛᵘ
  Π = kron(Φₛᵘ_normed, RBVars.Φₜᵘ)

  for (i, i_nₛ) = enumerate(snaps)
    println("Assembling generalized coordinate relative to snapshot $(i_nₛ), field u")
    S_i = RBVars.S.Sᵘ[:, (i_nₛ-1)*RBVars.Nₜ+1:i_nₛ*RBVars.Nₜ]
    û[:, i] = sum(Π_ij, dims=2) .* S_i

    #= for i_s = 1:RBVars.S.nₛᵘ
      for i_t = 1:RBVars.nₜᵘ
        Π_ij = reshape(Φₛᵘ_normed[:,i_s],:,1).*reshape(RBVars.Φₜᵘ[:,i_t],:,1)'
        û[index_mapping(i_s, i_t, RBVars), i] = sum(Π_ij .* S_i)
      end
    end
    sum(Π_ij .* S_i) =#
  end

  RBVars.S.û = û

  if RBInfo.S.save_offline_structures
    save_CSV(û, joinpath(RBInfo.S.paths.gen_coords_path, "û.csv"))
  end

end

function test_offline_phase(
  RBInfo::ROMInfoUnsteady{T},
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
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  var::String) where T

  if var == "M"
    println("The matrix $var is non-affine:
      running the MDEIM offline phase on $(RBInfo.S.nₛ_MDEIM) snapshots")
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
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  var::String) where T

  println("The vector $var is non-affine:
    running the DEIM offline phase on $(RBInfo.S.nₛ_MDEIM) snapshots")

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
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T}) where T

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
      save_CSV(l_val[i₁], joinpath(RBInfo.S.paths.ROM_structures_path,
        list_names[i₂]*".csv"))
    end
  end

  save_M_DEIM_structures(RBInfo.S, RBVars.S)

end

function set_operators(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T}) where T

  vcat(["M"], set_operators(RBInfo.S, RBVars.S))

end

function get_M_DEIM_structures(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T}) where T

  operators = String[]

  if RBInfo.S.probl_nl["M"]

    if isfile(joinpath(RBInfo.S.paths.ROM_structures_path, "MDEIMᵢ_M.csv"))
      println("Importing MDEIM offline structures for the mass matrix")
      RBVars.MDEIMᵢ_M = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.S.paths.ROM_structures_path,
        "MDEIMᵢ_M.csv"))
      RBVars.MDEIM_idx_M = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.S.paths.ROM_structures_path,
        "MDEIM_idx_M.csv"))[:]
      RBVars.sparse_el_M = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.S.paths.ROM_structures_path,
        "sparse_el_M.csv"))[:]
      RBVars.row_idx_M = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.S.paths.ROM_structures_path,
        "row_idx_M.csv"))[:]
      append!(operators, [])
    else
      println("Failed to import MDEIM offline structures for the mass matrix: must build them")
      append!(operators, ["M"])
    end

  end

  append!(operators, get_M_DEIM_structures(RBInfo.S, RBVars.S))

end

function get_offline_structures(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T}) where T

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars))
  append!(operators, get_M_DEIM_structures(RBInfo, RBVars))
  unique!(operators)

  operators

end

function get_θᵐ(
  FEMSpace₀::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  timesθ = get_timesθ(RBInfo)
  if !RBInfo.S.probl_nl["M"]
    θᵐ = [T(Param.mₜ(t_θ)) for t_θ = timesθ]
  else
    M_μ_sparse = build_sparse_mat(
      FEMSpace₀,FEMInfo,Param,RBVars.sparse_el_M,timesθ;var="M")
    θᵐ = (RBVars.MDEIMᵢ_M \
      Matrix{T}(reshape(M_μ_sparse, :, RBVars.Nₜ)[RBVars.MDEIM_idx_M, :]))
    #=
    Nₛᵘ = RBVars.S.Nₛᵘ
    θᵐ = zeros(RBVars.Qᵐ, RBVars.Nₜ)
    @simd for iₜ = 1:RBVars.Nₜ
      θᵐ[:,iₜ] = M_DEIM_online(M_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ],
        RBVars.MDEIMᵢ_M, RBVars.MDEIM_idx_M)
    end =#
  end

  reshape(θᵐ, RBVars.Qᵐ, RBVars.Nₜ)

end

function get_θᵐₛₜ(
  FEMSpace₀::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  if !RBInfo.S.probl_nl["M"]
    θᵐ = [one(T)]
  else
    timesθ_mod,MDEIM_idx_mod =
      modify_timesθ_and_MDEIM_idx(RBVars.MDEIM_idx_M,RBInfo,RBVars)
      M_μ_sparse = build_sparse_mat(FEMSpace₀, FEMInfo, Param, RBVars.sparse_el_M,
      timesθ_mod; var="M")
    θᵐ = RBVars.MDEIMᵢ_M\Vector(M_μ_sparse[MDEIM_idx_mod])
  end

  return θᵐ

end

function get_θᵃ(
  FEMSpace₀::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  timesθ = get_timesθ(RBInfo)
  if !RBInfo.S.probl_nl["A"]
    θᵃ = T.([Param.αₜ(t_θ,Param.μ) for t_θ = timesθ])
  else
    A_μ_sparse = build_sparse_mat(
      FEMSpace₀,FEMInfo,Param,RBVars.S.sparse_el_A,timesθ;var="A")
    θᵃ = (RBVars.S.MDEIMᵢ_A \
      Matrix{T}(reshape(A_μ_sparse, :, RBVars.Nₜ)[RBVars.S.MDEIM_idx_A, :]))
    #= Nₛᵘ = RBVars.S.Nₛᵘ
    θᵃ = zeros(RBVars.S.Qᵃ, RBVars.Nₜ)
    @simd for iₜ = 1:RBVars.Nₜ
      θᵃ[:,iₜ] = M_DEIM_online(A_μ_sparse[:,(iₜ-1)*Nₛᵘ+1:iₜ*Nₛᵘ],
        RBVars.S.MDEIMᵢ_A, RBVars.S.MDEIM_idx_A)
    end =#
  end

  reshape(θᵃ, RBVars.S.Qᵃ, RBVars.Nₜ)

end

function get_θᵃₛₜ(
  FEMSpace₀::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  if !RBInfo.S.probl_nl["A"]
    θᵃ = [1]
  else
    timesθ_mod,MDEIM_idx_mod =
      modify_timesθ_and_MDEIM_idx(RBVars.S.MDEIM_idx_A,RBInfo,RBVars)
    A_μ_sparse = build_sparse_mat(FEMSpace₀,FEMInfo, Param, RBVars.S.sparse_el_A,
      timesθ_mod; var="A")
    θᵃ = RBVars.S.MDEIMᵢ_A\Vector(A_μ_sparse[MDEIM_idx_mod])
  end

  return θᵃ

end

function get_θᶠʰ(
  FEMSpace₀::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  if RBInfo.build_Parametric_RHS
    error("Cannot fetch θᶠ, θʰ if the RHS is built online")
  end

  timesθ = get_timesθ(RBInfo)
  #θᶠ, θʰ = Float64[], Float64[]

  if !RBInfo.S.probl_nl["f"]
    θᶠ = T.([Param.fₜ(t_θ) for t_θ = timesθ])
  else
    F_μ = assemble_forcing(FEMSpace₀, RBInfo, Param)
    F = zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ)
    for (i,tᵢ) in enumerate(timesθ)
      F[:,i] = F_μ(tᵢ)
    end
    θᶠ = (RBVars.S.DEIMᵢ_F \ Matrix{T}(F_μ[RBVars.S.DEIM_idx_F, :]))
    #= @simd for iₜ = 1:RBVars.Nₜ
      append!(θᶠ,
        M_DEIM_online(F_μ(timesθ[iₜ]), RBVars.S.DEIMᵢ_F, RBVars.S.DEIM_idx_F))
    end =#
  end

  if !RBInfo.S.probl_nl["h"]
    θʰ = T.([Param.hₜ(t_θ) for t_θ = timesθ])
  else
    H_μ = assemble_neumann_datum(FEMSpace₀, RBInfo, Param)
    H = zeros(T, RBVars.S.Nₛᵘ, RBVars.Nₜ)
    for (i,tᵢ) in enumerate(timesθ)
      H[:,i] = H_μ(tᵢ)
    end
    θʰ = (RBVars.S.DEIMᵢ_H \ Matrix{T}(H_μ[RBVars.S.DEIM_idx_H, :]))
    #= @simd for iₜ = 1:RBVars.Nₜ
      append!(θʰ,
        M_DEIM_online(H_μ(timesθ[iₜ]), RBVars.S.DEIMᵢ_H, RBVars.S.DEIM_idx_H))
    end =#
  end

  reshape(θᶠ, RBVars.S.Qᶠ, RBVars.Nₜ), reshape(θʰ, RBVars.S.Qʰ, RBVars.Nₜ)

end

function get_θᶠʰₛₜ(
  FEMSpace₀::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  if RBInfo.S.build_Parametric_RHS
    error("Cannot fetch θᶠ, θʰ if the RHS is built online")
  end

  if !RBInfo.S.probl_nl["f"]
    θᶠ = [one(T)]
  else
    F_μ = assemble_forcing(FEMSpace₀, RBInfo, Param)
    _,DEIM_idx_mod = modify_timesθ_and_MDEIM_idx(RBVars.S.DEIM_idx_F,RBInfo,RBVars)
    θᶠ = T.(RBVars.S.DEIMᵢ_F\Vector(F_μ[DEIM_idx_mod]))
  end

  if !RBInfo.S.probl_nl["h"]
    θʰ = [one(T)]
  else
    H_μ = assemble_neumann_datum(FEMSpace₀, RBInfo, Param)
    _,DEIM_idx_mod = modify_timesθ_and_MDEIM_idx(RBVars.S.DEIM_idx_H,RBInfo,RBVars)
    θʰ = T.(RBVars.S.DEIMᵢ_H\Vector(H_μ[DEIM_idx_mod]))
  end

  return θᶠ,θʰ

end

function solve_RB_system(
  FEMSpace₀::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  get_RB_system(FEMSpace₀, RBInfo, RBVars, Param)

  println("Solving RB problem via backslash")
  println("Condition number of the system's matrix: $(cond(RBVars.S.LHSₙ[1]))")

  RBVars.S.online_time += @elapsed begin
    RBVars.S.uₙ = zeros(T, RBVars.nᵘ)
    @fastmath RBVars.S.uₙ = RBVars.S.LHSₙ[1] \ RBVars.S.RHSₙ[1]
  end

end

function reconstruct_FEM_solution(RBVars::PoissonUnsteady{T}) where T

  println("Reconstructing FEM solution from the newly computed RB one")
  uₙ = reshape(RBVars.S.uₙ, (RBVars.nₜᵘ, RBVars.S.nₛᵘ))
  @fastmath RBVars.S.ũ = RBVars.S.Φₛᵘ * (RBVars.Φₜᵘ * uₙ)'

end

function offline_phase(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T}) where T

  RBVars.Nₜ = convert(Int64, RBInfo.tₗ / RBInfo.δt)

  if RBInfo.S.import_snapshots
    get_snapshot_matrix(RBInfo, RBVars)
    import_snapshots_success = true
  else
    import_snapshots_success = false
  end

  if RBInfo.S.import_offline_structures
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

  if RBInfo.S.import_offline_structures
    operators = get_offline_structures(RBInfo, RBVars)
    if !isempty(operators)
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

end

function loop_on_params(
  FEMSpace₀::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  μ::Matrix{T},
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

    μ_nb = parse.(T, split(chop(μ[nb]; head=1, tail=1), ','))
    Param = get_ParamInfo(RBInfo, FEMInfo.problem_id, μ_nb)
    if RBInfo.perform_nested_POD
      nb_test = nb-90
      uₕ_test = Matrix{T}(CSV.read(joinpath(RBInfo.S.paths.FEM_snap_path,
      "uₕ_test.csv"), DataFrame))[:,(nb_test-1)*RBVars.Nₜ+1:nb_test*RBVars.Nₜ]
    else
      uₕ_test = Matrix{T}(CSV.read(joinpath(RBInfo.S.paths.FEM_snap_path, "uₕ.csv"),
      DataFrame))[:,(nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]
    end
    mean_uₕ_test += uₕ_test

    solve_RB_system(FEMSpace₀, RBInfo, RBVars, Param)
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

function online_phase(
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  μ::Matrix{T},
  param_nbs) where T

  model = DiscreteModelFromFile(RBInfo.S.paths.mesh_path)
  FEMSpace₀ = get_FEMSpace₀(FEMInfo.problem_id, FEMInfo, model)

  get_norm_matrix(RBInfo.S, RBVars.S)
  (ũ_μ,uₙ_μ,mean_uₕ_test,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,H1_L2_err,
    mean_online_time,mean_reconstruction_time) =
    loop_on_params(FEMSpace₀, RBInfo, RBVars, μ, param_nbs)

  adapt_time = 0.
  if RBInfo.adaptivity
    adapt_time = @elapsed begin
      (ũ_μ,uₙ_μ,_,mean_pointwise_err,mean_H1_err,mean_H1_L2_err,
      H1_L2_err,mean_online_time,mean_reconstruction_time) =
      adaptive_loop_on_params(FEMSpace₀, RBInfo, RBVars, mean_uₕ_test,
      mean_pointwise_err, μ, param_nbs)
    end
  end

  string_param_nbs = "Params"
  for Param_nb in param_nbs
    string_param_nbs *= "_" * string(Param_nb)
  end
  path_μ = joinpath(RBInfo.S.paths.results_path, string_param_nbs)

  if RBInfo.S.save_results
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

  if RBInfo.S.postprocess
    println("Post-processing the results...")
    post_process(RBInfo, pass_to_pp)
  end

  #=
  plot_stability_constant(FEMSpace,RBInfo,Param,Nₜ)
  =#

end

function post_process(RBInfo::UnsteadyInfo, d::Dict)
  if isfile(joinpath(RBInfo.S.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    MDEIM_Σ = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.S.paths.ROM_structures_path, "MDEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(MDEIM_Σ), MDEIM_Σ, "Decay singular values, MDEIM",
      ["σ"], "σ index", "σ value", RBInfo.S.paths.results_path; var="MDEIM_Σ")
  end
  if isfile(joinpath(RBInfo.S.paths.ROM_structures_path, "DEIM_Σ.csv"))
    DEIM_Σ = load_CSV(Matrix{T}(undef,0,0), joinpath( RBInfo.S.paths.ROM_structures_path, "DEIM_Σ.csv"))
    generate_and_save_plot(
      eachindex(DEIM_Σ), DEIM_Σ, "Decay singular values, DEIM",
      ["σ"], "σ index", "σ value", RBInfo.S.paths.results_path; var="DEIM_Σ")
  end

  times = collect(RBInfo.t₀+RBInfo.δt:RBInfo.δt:RBInfo.tₗ)
  FEMSpace = d["FEMSpace"]
  vtk_dir = joinpath(d["path_μ"], "vtk_folder")

  create_dir(vtk_dir)
  createpvd(joinpath(vtk_dir,"mean_point_err_u")) do pvd
    for (i,t) in enumerate(times)
      errₕt = FEFunction(FEMSpace.V(t), d["mean_point_err_u"][:,i])
      pvd[i] = createvtk(FEMSpace.Ω, joinpath(vtk_dir,
        "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
    end
  end

  generate_and_save_plot(times,d["mean_H1_err"],
    "Average ||uₕ(t) - ũ(t)||ₕ₁", ["H¹ err"], "time [s]", "H¹ error", d["path_μ"];
    var="H1_err")
  xvec = collect(eachindex(d["H1_L2_err"]))
  generate_and_save_plot(xvec,d["H1_L2_err"],
    "||uₕ - ũ||ₕ₁₋ₗ₂", ["H¹-l² err"], "Param μ number", "H¹-l² error", d["path_μ"];
    var="H1_L2_err")

  if length(keys(d)) == 8

    createpvd(joinpath(vtk_dir,"mean_point_err_p")) do pvd
      for (i,t) in enumerate(times)
        errₕt = FEFunction(FEMSpace.Q, d["mean_point_err_p"][:,i])
        pvd[i] = createvtk(FEMSpace.Ω, joinpath(vtk_dir,
          "mean_point_err_$i" * ".vtu"), cellfields = ["point_err" => errₕt])
      end
    end

    generate_and_save_plot(times,d["mean_L2_err"],
      "Average ||pₕ(t) - p̃(t)||ₗ₂", ["l² err"], "time [s]", "L² error", d["path_μ"];
      var="L2_err")
    xvec = collect(eachindex(d["L2_L2_err"]))
    generate_and_save_plot(xvec,d["L2_L2_err"],
      "||pₕ - p̃||ₗ₂₋ₗ₂", ["l²-l² err"], "Param μ number", "L²-L² error", d["path_μ"];
      var="L2_L2_err")

  end

end

function adaptive_loop_on_params(
  FEMSpace₀::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady{T},
  RBVars::PoissonUnsteady{T},
  mean_uₕ_test::Matrix{T},
  mean_pointwise_err::Matrix{T},
  μ::Matrix{T},
  param_nbs,
  n_adaptive=nothing) where T

  if isnothing(n_adaptive)
    nₛᵘ_add = floor(Int64,RBVars.S.nₛᵘ*0.1)
    nₜᵘ_add = floor(Int64,RBVars.nₜᵘ*0.1)
    n_adaptive = maximum(hcat([1,1],[nₛᵘ_add,nₜᵘ_add]),dims=2)
  end

  println("Running adaptive cycle: adding $n_adaptive temporal and spatial bases,
    respectively")

  time_err = zeros(RBVars.Nₜ)
  space_err = zeros(RBVars.S.Nₛᵘ)
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
    Sᵘ = Matrix{T}(CSV.read(joinpath(RBInfo.S.paths.FEM_snap_path, "uₕ.csv"),
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

  loop_on_params(FEMSpace₀,RBInfo,RBVars,μ,param_nbs)

end

function plot_stability_constants(
  FEMSpace::FEMProblem,
  RBInfo::ROMInfoUnsteady{T},
  Param::ParametricInfoUnsteady{D,T}) where {D,T}

  M = assemble_mass(FEMSpace, RBInfo, Param)(0.0)
  A = assemble_stiffness(FEMSpace, RBInfo, Param)(0.0)
  stability_constants = []
  for Nₜ = 10:10:1000
    const_Nₜ = compute_stability_constant(RBInfo,Nₜ,M,A)
    append!(stability_constants, const_Nₜ)
  end
  p = Plot.plot(collect(10:10:1000),
    stability_constants, xaxis=:log, yaxis=:log, lw = 3,
    label="||(Aˢᵗ)⁻¹||₂", title = "Euclidean norm of (Aˢᵗ)⁻¹", legend=:topleft)
  p = Plot.plot!(collect(10:10:1000), collect(10:10:1000),
    xaxis=:log, yaxis=:log, lw = 3, label="Nₜ")
  xlabel!("Nₜ")
  savefig(p, joinpath(RBInfo.S.paths.results_path, "stability_constant.eps"))

  function compute_stability_constant(RBInfo,Nₜ,M,A)
    δt = RBInfo.tₗ/Nₜ
    B₁ = RBInfo.θ*(M + RBInfo.θ*δt*A)
    B₂ = RBInfo.θ*(-M + (1-RBInfo.θ)*δt*A)
    λ₁,_ = eigs(B₁)
    λ₂,_ = eigs(B₂)
    return 1/(minimum(abs.(λ₁)) + minimum(abs.(λ₂)))
  end

end
