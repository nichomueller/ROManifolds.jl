################################# OFFLINE ######################################

function check_norm_matrix(RBVars::StokesST)
  check_norm_matrix(RBVars.Steady)
end

function PODs_space(
  RBInfo::Info,
  RBVars::StokesST)

  PODs_space(RBInfo, RBVars.Poisson)

  println("Performing the spatial POD for field p, using a tolerance of $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars.Steady)
  RBVars.Φₛᵖ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.Xᵖ₀)
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function PODs_time(
  RBInfo::ROMInfoST,
  RBVars::StokesST{T}) where T

  PODs_time(RBInfo, RBVars.Poisson)

  println("Performing the temporal POD for field p, using a tolerance of $(RBInfo.ϵₜ)")

  if RBInfo.time_reduction_technique == "ST-HOSVD"
    Sᵖ = RBVars.Φₛᵖ' * RBVars.Sᵖ
  else
    Sᵖ = RBVars.Sᵖ
  end
  Sᵖₜ = mode₂_unfolding(Sᵖ, RBInfo.nₛ)

  Φₜᵖ = POD(Sᵖₜ, RBInfo.ϵₜ)
  RBVars.Φₜᵖ = Φₜᵖ
  RBVars.nₜᵖ = size(Φₜᵖ)[2]

end

function time_supremizers(Φₜᵘ::Matrix{T}, Φₜᵖ::Matrix{T}) where T

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

  ΦₜᵘΦₜᵖ = Φₜᵘ' * Φₜᵖ
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
      Φₜᵖ_l_on_Φₜᵘ = compute_projection_on_span(Φₜᵖ[:, l], Φₜᵘ)
      Φₜᵘ_to_add = ((Φₜᵖ[:, l] - Φₜᵖ_l_on_Φₜᵘ) / norm(Φₜᵖ[:, l] - Φₜᵖ_l_on_Φₜᵘ))
      Φₜᵘ = hcat(Φₜᵘ, Φₜᵘ_to_add)
      ΦₜᵘΦₜᵖ = hcat(ΦₜᵘΦₜᵖ, Φₜᵘ_to_add' * Φₜᵖ)
    end

  end

  Φₜᵘ

end

function supr_enrichment_time(
  RBVars::StokesST)

  RBVars.Φₜᵘ = time_supremizers(RBVars.Φₜᵘ, RBVars.Φₜᵖ)
  RBVars.nₜᵘ = size(RBVars.Φₜᵘ)[2]

end

function index_mapping(i::Int, j::Int, RBVars::StokesST, var="u")

  if var == "u"
    return index_mapping(i, j, RBVars.Poisson)
  elseif var == "p"
    return Int((i-1) * RBVars.nₜᵖ + j)
  else
    error("Unrecognized variable")
  end

end

function get_generalized_coordinates(
  RBInfo::ROMInfoST,
  RBVars::StokesST{T},
  snaps::Vector{Int}) where T

  if check_norm_matrix(RBVars)
    get_norm_matrix(RBInfo, RBVars)
  end

  get_generalized_coordinates(RBInfo, RBVars.Poisson)

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
    save_CSV(p̂, joinpath(RBInfo.ROM_structures_path, "p̂.csv"))
  end

end

function set_operators(
  RBInfo::Info,
  RBVars::StokesST)

  append!(["M"], set_operators(RBInfo, RBVars.Steady))

end

function get_A(
  RBInfo::Info,
  RBVars::StokesST)

  get_A(RBInfo, RBVars.Poisson)

end

function get_M(
  RBInfo::ROMInfoST,
  RBVars::StokesST)

  get_M(RBInfo, RBVars.Poisson)

end

function get_B(
  RBInfo::Info,
  RBVars::StokesST)

  get_B(RBInfo, RBVars.Steady)

end

function get_F(
  RBInfo::Info,
  RBVars::StokesST)

  get_F(RBInfo, RBVars.Poisson)

end

function get_H(
  RBInfo::Info,
  RBVars::StokesST)

  get_H(RBInfo, RBVars.Poisson)

end

function get_L(
  RBInfo::Info,
  RBVars::StokesST)

  get_L(RBInfo, RBVars.Poisson)

end

function get_Lc(
  RBInfo::Info,
  RBVars::StokesST)

  get_Lc(RBInfo, RBVars.Steady)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::StokesST{T},
  var::String) where T

  if var == "B"
    println("Assembling affine primal operator B")
    B = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))
    RBVars.Bₙ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ, 1)
    RBVars.Bₙ[:,:,1] = (RBVars.Φₛᵖ)' * B * RBVars.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoST,
  RBVars::StokesST,
  var::String)

  println("The matrix $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
  if var == "A"
    if isempty(RBVars.MDEIM_mat_A)
      (RBVars.MDEIM_mat_A, RBVars.MDEIM_idx_A, RBVars.MDEIMᵢ_A,
      RBVars.row_idx_A,RBVars.sparse_el_A) = MDEIM_offline(RBInfo, RBVars, "A")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_A, RBVars.row_idx_A, var)
  elseif var == "B"
    if isempty(RBVars.MDEIM_mat_B)
      (RBVars.MDEIM_mat_B, RBVars.MDEIM_idx_B, RBVars.MDEIMᵢ_B,
      RBVars.row_idx_B,RBVars.sparse_el_B) = MDEIM_offline(RBInfo, RBVars, "B")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_B, RBVars.row_idx_B, var)
  elseif var == "M"
    if isempty(RBVars.MDEIM_mat_M)
      (RBVars.MDEIM_mat_M, RBVars.MDEIM_idx_M, RBVars.MDEIMᵢ_M,
      RBVars.row_idx_M,RBVars.sparse_el_M) = MDEIM_offline(RBInfo, RBVars, "M")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_M, RBVars.row_idx_M, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesST,
  MDEIM_mat::Matrix,
  row_idx::Vector{Int},
  var::String)

  if var == "B"
    Q = size(MDEIM_mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.Nₛᵖ)
    MatqΦ = zeros(T,RBVars.Nₛᵖ,RBVars.nₛᵘ,Q)
    @simd for j = 1:RBVars.Nₛᵖ
      Mat_idx = findall(x -> x == j, r_idx)
      MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
    end

    Matₙ = reshape(RBVars.Φₛᵖ' *
      reshape(MatqΦ,RBVars.Nₛᵖ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}
    RBVars.Bₙ = Matₙ
    RBVars.Qᵇ = Q

  else
    assemble_reduced_mat_MDEIM(RBVars.Poisson, MDEIM_mat, row_idx, var)
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::StokesST,
  var::String)

  assemble_affine_vectors(RBInfo, RBVars.Steady, var)

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoST,
  RBVars::StokesST,
  var::String)

  assemble_DEIM_vectors(RBInfo, RBVars.Poisson, var)

end

function assemble_reduced_mat_DEIM(
  RBInfo::ROMInfoST,
  RBVars::StokesST,
  DEIM_mat::Matrix,
  var::String)

  assemble_reduced_mat_DEIM(RBInfo, RBVars.Steady, DEIM_mat, var)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::PoissonS)

  affine_vars = (reshape(RBVars.Bₙ, :, RBVars.Qᵇ)::Matrix{T}, RBVars.Lcₙ)
  affine_names = ("Bₙ", "Lcₙ")
  save_structures_in_list(affine_vars, affine_names, RBInfo.ROM_structures_path)

  save_affine_structures(RBInfo, RBVars.Poisson)

end

function save_M_DEIM_structures(
  RBInfo::Info,
  RBVars::PoissonS)

  M_DEIM_vars = (
    RBVars.MDEIM_mat_B, RBVars.MDEIMᵢ_B, RBVars.MDEIM_idx_B, RBVars.row_idx_B,
    RBVars.sparse_el_B, RBVars.DEIM_mat_Lc, RBVars.DEIMᵢ_Lc, RBVars.DEIM_idx_Lc,)
    RBVars.sparse_el_Lc
  M_DEIM_names = (
    "MDEIM_mat_B","MDEIMᵢ_B","MDEIM_idx_B","row_idx_B","sparse_el_B",
    "DEIM_mat_Lc","DEIMᵢ_Lc","DEIM_idx_Lc","sparse_el_Lc")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path)

  save_M_DEIM_structures(RBInfo, RBVars.Poisson)

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::Info,
  RBVars::StokesST,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int})

  get_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::StokesST,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  save_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks, operators)

end

function get_θ_matrix(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::StokesS,
  Param::ParamInfoS,
  var::String)

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.α, RBVars.MDEIMᵢ_A,
      RBVars.MDEIM_idx_A, RBVars.sparse_el_A, RBVars.MDEIM_idx_time_A, "A")::Matrix{T}
  elseif var == "B"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.b, RBVars.MDEIMᵢ_B,
      RBVars.MDEIM_idx_B, RBVars.sparse_el_B, RBVars.MDEIM_idx_time_B, "B")::Matrix{T}
  elseif var == "M"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.m, RBVars.MDEIMᵢ_M,
      RBVars.MDEIM_idx_M, RBVars.sparse_el_M, RBVars.MDEIM_idx_time_M, "M")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_θ_vector(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::StokesS,
  Param::ParamInfoS,
  var::String)

  if var == "F"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param, Param.f, RBVars.DEIMᵢ_F,
      RBVars.DEIM_idx_F, RBVars.sparse_el_F, RBVars.DEIM_idx_time_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param, Param.h, RBVars.DEIMᵢ_H,
      RBVars.DEIM_idx_H, RBVars.sparse_el_H, RBVars.DEIM_idx_time_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.DEIMᵢ_L,
      RBVars.DEIM_idx_L, RBVars.sparse_el_L, RBVars.DEIM_idx_time_L, "L")::Matrix{T}
  elseif var == "Lc"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.DEIMᵢ_Lc,
      RBVars.DEIM_idx_Lc, RBVars.sparse_el_Lc, RBVars.DEIM_idx_time_Lc, "Lc")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_Q(
  RBInfo::Info,
  RBVars::StokesST)

  if RBVars.Qᵇ == 0
    RBVars.Qᵇ = size(RBVars.Bₙ)[end]
  end
  if !RBInfo.online_RHS
    if RBVars.Qˡᶜ == 0
      RBVars.Qˡᶜ = size(RBVars.Lcₙ)[end]
    end
  end

  get_Q(RBInfo, RBVars.Poisson)

end

function assemble_param_RHS(
  FEMSpace::FEMProblemST,
  RBInfo::Info,
  RBVars::StokesST,
  Param::ParamInfoST)

  assemble_param_RHS(FEMSpace, RBInfo, RBVars.Poisson, Param)

  Lc_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "Lc")

  RHS_c = zeros(T, RBVars.Nₛᵖ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo)

  for (i,tᵢ) in enumerate(timesθ)
    RHS_c[:, i] = - Lc_t(tᵢ)
  end

  RHS_cₙ = RBVars.Φₛᵘ' * (RHS_c * RBVars.Φₜᵘ)
  push!(RBVars.RHSₙ, reshape(RHS_cₙ', :, 1))::Vector{Matrix{T}}

end

function adaptive_loop_on_params(
  FEMSpace::FEMProblemST,
  RBInfo::ROMInfoST,
  RBVars::StokesST{T},
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
