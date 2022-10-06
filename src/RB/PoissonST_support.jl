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

  if isempty(RBVars.Xᵘ₀)
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

  op = get_A(RBInfo, RBVars.Steady)

  if isempty(op)
    if "A" ∈ RBInfo.probl_nl
      if isfile(joinpath(RBInfo.ROM_structures_path, "time_idx_A.csv"))
        RBVars.MDEIM_A.time_idx = load_CSV(Vector{Int}(undef,0),
          joinpath(RBInfo.ROM_structures_path, "time_idx_A.csv"))
      else
        op = ["A"]
      end
    end
  end

  op

end

function get_M(
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))

    Mₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Mₙ.csv"))
    RBVars.Mₙ = reshape(Mₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}

    if "M" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_M.Matᵢ, RBVars.MDEIM_M.idx, RBVars.MDEIM_M.time_idx, RBVars.MDEIM_M.el) =
        load_structures_in_list(("Matᵢ_M", "idx_M", "time_idx_M", "el_M"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0),
        Vector{Int}(undef,0)), RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for M: must build them")
    op = ["M"]

  end

  op

end

function get_F(
  RBInfo::Info,
  RBVars::PoissonST)

  op = get_F(RBInfo, RBVars.Steady)

  if isempty(op)
    if "F" ∈ RBInfo.probl_nl
      if isfile(joinpath(RBInfo.ROM_structures_path, "time_idx_F.csv"))
        RBVars.MDEIM_F.time_idx = load_CSV(Vector{Int}(undef,0),
          joinpath(RBInfo.ROM_structures_path, "time_idx_F.csv"))
      else
        op = ["F"]
      end
    end
  end

  op

end

function get_H(
  RBInfo::Info,
  RBVars::PoissonST)

  op = get_H(RBInfo, RBVars.Steady)

  if isempty(op)
    if "H" ∈ RBInfo.probl_nl
      if isfile(joinpath(RBInfo.ROM_structures_path, "time_idx_H.csv"))
        RBVars.MDEIM_H.time_idx = load_CSV(Vector{Int}(undef,0),
          joinpath(RBInfo.ROM_structures_path, "time_idx_H.csv"))
      else
        op = ["H"]
      end
    end
  end

  op

end

function get_L(
  RBInfo::Info,
  RBVars::PoissonST)

  op = get_L(RBInfo, RBVars.Steady)

  if isempty(op)
    if "L" ∈ RBInfo.probl_nl
      if isfile(joinpath(RBInfo.ROM_structures_path, "time_idx_L.csv"))
        RBVars.MDEIM_L.time_idx = load_CSV(Vector{Int}(undef,0),
          joinpath(RBInfo.ROM_structures_path, "time_idx_L.csv"))
      else
        op = ["L"]
      end
    end
  end

  op

end

function assemble_affine_structures(
  RBInfo::Info,
  RBVars::PoissonST{T},
  var::String) where T

  if var == "M"
    println("Assembling affine reduced M")
    M = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "M.csv"))
    RBVars.Mₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
    RBVars.Mₙ[:,:,1] = (RBVars.Φₛᵘ)' * M * RBVars.Φₛᵘ
  else
    assemble_affine_structures(RBInfo, RBVars.Steady, var)
  end

end

function assemble_MDEIM_structures(
  RBInfo::ROMInfoST,
  RBVars::PoissonST,
  var::String)

  println("The variable $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "A"
    if isempty(RBVars.MDEIM_A.Mat)
      MDEIM_offline!(RBVars.MDEIM_A, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_A, var)
  elseif var == "M"
    if isempty(RBVars.MDEIM_M.Mat)
      MDEIM_offline!(RBVars.MDEIM_M, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_M, var)
  elseif var == "F"
    if isempty(RBVars.MDEIM_F.Mat)
      MDEIM_offline!(RBVars.MDEIM_F, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_F, var)
  elseif var == "H"
    if isempty(RBVars.MDEIM_H.Mat)
      MDEIM_offline!(RBVars.MDEIM_H, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_H, var)
  elseif var == "L"
    if isempty(RBVars.MDEIM_L.Mat)
      MDEIM_offline!(RBVars.MDEIM_L, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_L, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonST{T},
  MDEIM::MDEIMm,
  var::String) where T

  Q = size(MDEIM.Mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, RBVars.Nₛᵘ)

  function assemble_ith_row(i::Int)
    Mat_idx = findall(x -> x == i, r_idx)
    Matrix(reshape((MDEIM.Mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])', 1, :))
  end

  VecMatΦ = Broadcasting(assemble_ith_row)(1:RBVars.Nₛᵘ)::Vector{Matrix{T}}
  MatqΦ = Matrix{T}(reduce(vcat, VecMatΦ))::Matrix{T}
  Matₙ = Matrix{T}(reshape(RBVars.Φₛᵘ' * MatqΦ, RBVars.nₛᵘ, :, Q))::Matrix{T}

  if var == "M"
    RBVars.Mₙ = Matₙ
  else
    RBVars.Aₙ = Matₙ
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonST{T},
  MDEIM::MDEIMv,
  var::String) where T

  assemble_reduced_mat_MDEIM(RBVars.Steady, MDEIM, var)

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::PoissonST{T},
  operators::Vector{String}) where T

  affine_vars = (reshape(RBVars.Mₙ, RBVars.nₛᵘ ^ 2, :)::Matrix{T},)
  affine_names = ("Mₙ",)
  affine_entry = get_affine_entries(operators, affine_names)
  save_structures_in_list(affine_vars[affine_entry], affine_names[affine_entry],
    RBInfo.ROM_structures_path)

  M_DEIM_vars = (
    RBVars.MDEIM_M.Matᵢ, RBVars.MDEIM_M.idx, RBVars.MDEIM_M.time_idx, RBVars.MDEIM_M.el,
    RBVars.MDEIM_A.time_idx, RBVars.MDEIM_F.time_idx, RBVars.MDEIM_H.time_idx,
    RBVars.MDEIM_L.time_idx)
  M_DEIM_names = (
    "Matᵢ_M","idx_M","time_idx_M","el_M",
    "time_idx_A","time_idx_F","time_idx_H","time_idx_L")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path)

  save_assembled_structures(RBInfo, RBVars.Steady, operators)

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
  Param::ParamInfoST,
  var::String) where T

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.α, RBVars.MDEIM_A, "A")::Matrix{T}
  elseif var == "M"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.m, RBVars.MDEIM_M, "M")::Matrix{T}
  elseif var == "F"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.f, RBVars.MDEIM_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.h, RBVars.MDEIM_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.MDEIM_L, "L")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end


function assemble_param_RHS(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::PoissonST{T},
  Param::ParamInfoST) where T

  println("Assembling RHS exactly using θ-method time scheme, θ=$(RBInfo.θ)")

  F_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")

  RHS = zeros(T, RBVars.Nₛᵘ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo)

  for (i,tᵢ) in enumerate(timesθ)
    RHS[:,i] = F_t(tᵢ) + H_t(tᵢ) - L_t(tᵢ)
  end

  RHSₙ = RBVars.Φₛᵘ'*(RHS*RBVars.Φₜᵘ)
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
