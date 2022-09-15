################################# OFFLINE ######################################

PODs_space(RBInfo::Info, RBVars::PoissonST) =
  PODs_space(RBInfo, RBVars.Steady)

function PODs_time(
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T}) where T

  println("Performing the temporal POD for field u, using a tolerance of $(RBInfo.ϵₜ)")

  if RBInfo.time_reduction_technique == "ST-HOSVD"
    Sᵘ = RBVars.Φₛᵘ' * RBVars.Sᵘ
  else
    Sᵘ = RBVars.Sᵘ
  end
  Sᵘₜ = mode₂_unfolding(Sᵘ, RBInfo.nₛ)

  Φₜᵘ = POD(Sᵘₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵘ = Φₜᵘ
  RBVars.nₜᵘ = size(Φₜᵘ)[2]

end

function index_mapping(
  i::Int,
  j::Int,
  RBVars::PoissonST)

  Int((i-1)*RBVars.nₜᵘ+j)

end

function get_generalized_coordinates(
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  snaps::Vector{Int}) where T

  if check_norm_matrix(RBVars.Steady)
    get_norm_matrix(RBInfo, RBVars.Steady)
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
    save_CSV(û, joinpath(RBInfo.ROM_structures_path, "û.csv"))
  end

end

function set_operators(
  RBInfo::Info,
  RBVars::PoissonST)

  vcat(["M"], set_operators(RBInfo, RBVars.Steady))

end

function get_A(
  RBInfo::Info,
  RBVars::PoissonST)

  if "A" ∈ RBInfo.probl_nl
    if isfile(joinpath(RBInfo.ROM_structures_path, "MDEIM_idx_time_A.csv"))
      RBVars.MDEIM_idx_time_A = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.ROM_structures_path, "MDEIM_idx_time_A.csv"))
    end
  end

  get_A(RBInfo, RBVars.Steady)

end

function get_M(
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T}) where T

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
      return [""]
    else
      println("Failed to import MDEIM offline structures for M: must build them")
      return ["M"]
    end

  else

    if isfile(joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))
      println("Importing reduced affine mass matrix")
      Mₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))
      RBVars.Mₙ = reshape(Mₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}
      RBVars.Qᵐ = size(RBVars.Mₙ)[end]
      return [""]
    else
      println("Failed to import the reduced affine M: must build it")
      return ["M"]
    end

  end

end

function get_F(
  RBInfo::Info,
  RBVars::PoissonST)

  if "F" ∈ RBInfo.probl_nl
    if isfile(joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_F.csv"))
      RBVars.DEIM_idx_time_F = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_F.csv"))
    end
  end

  get_F(RBInfo, RBVars.Steady)

end

function get_H(
  RBInfo::Info,
  RBVars::PoissonST)

  if "H" ∈ RBInfo.probl_nl
    if isfile(joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_H.csv"))
      RBVars.DEIM_idx_time_H = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_H.csv"))
    end
  end

  get_H(RBInfo, RBVars.Steady)

end

function get_L(
  RBInfo::Info,
  RBVars::PoissonST)

  if "L" ∈ RBInfo.probl_nl
    if isfile(joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_L.csv"))
      RBVars.DEIM_idx_time_L = load_CSV(Vector{Int}(undef,0),
        joinpath(RBInfo.ROM_structures_path, "DEIM_idx_time_L.csv"))
    end
  end

  get_L(RBInfo, RBVars.Steady)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::PoissonST{T},
  var::String) where T

  if var == "M"
    RBVars.Qᵐ = 1
    println("Assembling affine reduced mass")
    M = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "M.csv"))
    RBVars.Mₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
    RBVars.Mₙ[:,:,1] = (RBVars.Φₛᵘ)' * M * RBVars.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.Steady, var)
  end

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoST,
  RBVars::PoissonST,
  var::String)

  if var == "M"
    println("The matrix $var is non-affine:
      running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
    if isempty(RBVars.MDEIM_mat_M)
      (RBVars.MDEIM_mat_M, RBVars.MDEIM_idx_M, RBVars.MDEIMᵢ_M, RBVars.row_idx_M,
        RBVars.sparse_el_M, RBVars.MDEIM_idx_time_M) = MDEIM_offline(RBInfo, RBVars, "M")
    end
    assemble_reduced_mat_MDEIM(
      RBVars,RBVars.MDEIM_mat_M,RBVars.row_idx_M,"M")
  elseif var == "A"
    if isempty(RBVars.MDEIM_mat_A)
      (RBVars.MDEIM_mat_A, RBVars.MDEIM_idx_A, RBVars.MDEIMᵢ_A,
      RBVars.row_idx_A,RBVars.sparse_el_A, RBVars.MDEIM_idx_time_A) = MDEIM_offline(RBInfo, RBVars, "A")
    end
    assemble_reduced_mat_MDEIM(
      RBVars,RBVars.MDEIM_mat_A,RBVars.row_idx_A,"A")
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonST{T},
  MDEIM_mat::Matrix{T},
  row_idx::Vector{Int},
  var::String) where T

  Q = size(MDEIM_mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.Nₛᵘ)
  MatqΦ = zeros(T, RBVars.Nₛᵘ,RBVars.nₛᵘ,Q)
  @simd for j = 1:RBVars.Nₛᵘ
    Mat_idx = findall(x -> x == j, r_idx)
    MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
  end
  Matₙ = reshape(RBVars.Φₛᵘ' *
    reshape(MatqΦ,RBVars.Nₛᵘ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}

  if var == "M"
    RBVars.Mₙ = Matₙ
    RBVars.Qᵐ = Q
  else
    RBVars.Aₙ = Matₙ
    RBVars.Qᵃ = Q
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::PoissonST{T},
  var::String) where T

  assemble_affine_vectors(RBInfo, RBVars.Steady, var)

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoST,
  RBVars::PoissonST,
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
  elseif var == "L"
    if isempty(RBVars.DEIM_mat_L)
      RBVars.DEIM_mat_L, RBVars.DEIM_idx_L, RBVars.DEIMᵢ_L, RBVars.sparse_el_L =
        DEIM_offline(RBInfo,"L")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_L,"L")
  else
    error("Unrecognized variable on which to perform DEIM")
  end

end

function assemble_reduced_mat_DEIM(
  RBVars::PoissonST{T},
  DEIM_mat::Matrix{T},
  var::String) where T

  Q = size(DEIM_mat)[2]
  Vecₙ = zeros(T, RBVars.nₛᵘ,1,Q)
  @simd for q = 1:Q
    Vecₙ[:,:,q] = RBVars.Φₛᵘ' * Vector{T}(DEIM_mat[:, q])
  end
  Vecₙ = reshape(Vecₙ,:,Q)::Matrix{T}

  if var == "F"
    RBVars.Fₙ = Vecₙ
    RBVars.Qᶠ = Q
  elseif var == "H"
    RBVars.Hₙ = Vecₙ
    RBVars.Qʰ = Q
  elseif var == "L"
    RBVars.Lₙ = Vecₙ
    RBVars.Qˡ = Q
  else
    error("Unrecognized vector to assemble with DEIM")
  end

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::PoissonST)

  affine_vars = (reshape(RBVars.Mₙ, :, RBVars.Qᵐ)::Matrix{T},)
  affine_names = ("Mₙ",)
  save_structures_in_list(affine_vars, affine_names, RBInfo.ROM_structures_path)

  M_DEIM_vars = (RBVars.MDEIM_mat_M, RBVars.MDEIMᵢ_M, RBVars.MDEIM_idx_M,
    RBVars.sparse_el_M, RBVars.row_idx_M, RBVars.MDEIM_idx_time_A,
    RBVars.MDEIM_idx_time_M, RBVars.DEIM_idx_time_F, RBVars.DEIM_idx_time_H)
  M_DEIM_names = ("MDEIM_mat_M", "MDEIMᵢ_M", "MDEIM_idx_M", "sparse_el_M",
   "row_idx_M", "MDEIM_idx_time_A", "MDEIM_idx_time_M", "DEIM_idx_time_F", "DEIM_idx_time_H")
  save_structures_in_list(list_M_DEIM, list_names, RBInfo.ROM_structures_path)

  save_assembled_structures(RBInfo, RBVars.Steady)

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::Info,
  RBVars::PoissonST,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int})

  get_system_blocks(RBInfo, RBVars.Steady, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::PoissonST,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  save_system_blocks(RBInfo, RBVars.Steady, LHS_blocks, RHS_blocks, operators)

end

function get_θ_matrix(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  Param::UnsteadyParametricInfo,
  var::String) where T

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param.α, RBVars.MDEIMᵢ_A,
      RBVars.MDEIM_idx_A, RBVars.sparse_el_A, RBVars.MDEIM_idx_time_A, "A")::Matrix{T}
  elseif var == "M"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param.m, RBVars.MDEIMᵢ_M,
      RBVars.MDEIM_idx_M, RBVars.sparse_el_M, RBVars.MDEIM_idx_time_M, "M")::Matrix{T}
  else
    error("Unrecognized variable")
  end

  θ::Matrix{T}

end

function get_θ_vector(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  Param::UnsteadyParametricInfo,
  var::String) where T

  if var == "F"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.f, RBVars.DEIMᵢ_F,
      RBVars.DEIM_idx_F, RBVars.sparse_el_F, RBVars.DEIM_idx_time_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.h, RBVars.DEIMᵢ_H,
      RBVars.DEIM_idx_H, RBVars.sparse_el_H, RBVars.DEIM_idx_time_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.g, RBVars.DEIMᵢ_L,
      RBVars.DEIM_idx_L, RBVars.sparse_el_L, RBVars.DEIM_idx_time_L, "L")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_Q(
  RBInfo::Info,
  RBVars::PoissonST)

  if RBVars.Qᵐ == 0
    RBVars.Qᵐ = size(RBVars.Mₙ)[end]
  end

  get_Q(RBInfo, RBVars.Steady)

end

function assemble_param_RHS(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  Param::UnsteadyParametricInfo) where T

  println("Assembling RHS exactly using θ-method time scheme, θ=$(RBInfo.θ)")

  F_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")

  RHS = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo)

  for (i,tᵢ) in enumerate(timesθ)
    RHS[:,i] = F_t(tᵢ) + H_t(tᵢ) - L_t(tᵢ)
  end

  RHSₙ = RBInfo.δt*RBInfo.θ * RBVars.Φₛᵘ'*(RHS*RBVars.Φₜᵘ)
  push!(RBVars.RHSₙ, reshape(RHSₙ',:,1))::Vector{Matrix{T}}

end

function adaptive_loop_on_params(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
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
