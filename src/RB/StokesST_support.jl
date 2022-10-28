################################# OFFLINE ######################################
function POD_space(
  RBInfo::Info,
  RBVars::StokesST)

  POD_space(RBInfo, RBVars.Poisson)

  println("Spatial POD for field p, tolerance: $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars.Steady)
  RBVars.Φₛᵖ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.Xp₀)
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function supr_enrichment_space(
  RBInfo::Info,
  RBVars::StokesST)

  supr_primal = primal_supremizers(RBInfo, RBVars.Steady)
  RBVars.Φₛ = hcat(RBVars.Φₛ, supr_primal)
  RBVars.nₛᵘ = size(RBVars.Φₛ)[2]

end

function POD_time(
  RBInfo::ROMInfoST,
  RBVars::StokesST{T}) where T

  POD_time(RBInfo, RBVars.Poisson)

  println("Temporal POD for field p, tolerance: $(RBInfo.ϵₜ)")

  if RBInfo.t_red_method == "ST-HOSVD"
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

  function projection_on_current_space(
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
  count = 0

  for l = 1:size(ΦₜᵘΦₜᵖ)[2]

    if l == 1
      ξ[:,l] = ΦₜᵘΦₜᵖ[:,1]
      enrich = (norm(ξ[:,l]) ≤ 1e-2)
    else
      ξ[:,l] = projection_on_current_space(ΦₜᵘΦₜᵖ[:, l], ΦₜᵘΦₜᵖ[:, 1:l-1])
      enrich = (norm(ξ[:,l] - ΦₜᵘΦₜᵖ[:,l]) ≤ 1e-2)
    end

    if enrich
      Φₜᵖ_l_on_Φₜᵘ = projection_on_current_space(Φₜᵖ[:, l], Φₜᵘ)
      Φₜᵘ_to_add = ((Φₜᵖ[:, l] - Φₜᵖ_l_on_Φₜᵘ) / norm(Φₜᵖ[:, l] - Φₜᵖ_l_on_Φₜᵘ))
      Φₜᵘ = hcat(Φₜᵘ, Φₜᵘ_to_add)
      ΦₜᵘΦₜᵖ = hcat(ΦₜᵘΦₜᵖ, Φₜᵘ_to_add' * Φₜᵖ)
      count += 1
    end

  end

  println("Added $count time supremizers to Φₜᵘ; final nₜᵘ is: $(size(Φₜᵘ)[1])")
  Φₜᵘ

end

function supr_enrichment_time(
  RBVars::StokesST)

  RBVars.Φₜᵘ = time_supremizers(RBVars.Φₜᵘ, RBVars.Φₜᵖ)
  RBVars.nₜᵘ = size(RBVars.Φₜᵘ)[2]

end

function index_mapping(i::Int, j::Int, RBVars::StokesST, var="u")

  if var == "u"
    index_mapping(i, j, RBVars.Poisson)
  elseif var == "p"
    Int((i-1) * RBVars.nₜᵖ + j)
  else
    error("Unrecognized variable")
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
  RBVars::StokesST{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))

    Bₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))
    RBVars.Bₙ = reshape(Bₙ, RBVars.nₛᵖ, RBVars.nₛᵘ, :)::Array{T,3}

    if "B" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_B.Matᵢ, RBVars.MDEIM_B.idx, RBVars.MDEIM_B.el) =
        load_structures_in_list(("Matᵢ_B", "idx_B", "el_B"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for B: must build them")
    op = ["B"]

  end

  op

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
  RBVars::StokesST{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Lcₙ.csv"))

    RBVars.Lcₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Lcₙ.csv"))

    if "Lc" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_Lc.Matᵢ, RBVars.MDEIM_Lc.idx, RBVars.MDEIM_Lc.el) =
        load_structures_in_list(("Matᵢ_Lc", "idx_Lc", "el_Lc"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for Lc: must build them")
    op = ["Lc"]

  end

  op

end

function assemble_affine_structures(
  RBInfo::Info,
  RBVars::StokesST{T},
  var::String) where T

  if var == "B"
    println("Assembling affine reduced B")
    B = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))
    RBVars.Bₙ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ, 1)
    RBVars.Bₙ[:,:,1] = (RBVars.Φₛᵖ)' * B * RBVars.Φₛ
  elseif var == "Lc"
    println("Assembling affine reduced Lc")
    Lc = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_structures_path(RBInfo), "Lc.csv"))
    RBVars.Lcₙ = RBVars.Φₛᵖ' * Lc
  else
    assemble_affine_structures(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_MDEIM_structures(
  RBInfo::ROMInfoST,
  RBVars::StokesST,
  var::String)

  println("The variable  $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "A"
    if isempty(RBVars.MDEIM_A.Mat)
      MDEIM_offline!(RBVars.MDEIM_A, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_A, var)
  elseif var == "B"
    if isempty(RBVars.MDEIM_B.Mat)
      MDEIM_offline!(RBVars.MDEIM_B, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_B, var)
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
  elseif var == "Lc"
    if isempty(RBVars.MDEIM_Lc.Mat)
      MDEIM_offline!(RBVars.MDEIM_Lc, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_Lc, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesST,
  MDEIM::MMDEIM,
  var::String)

  if var == "B"
    Q = size(MDEIM.Mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, RBVars.Nₛᵖ)
    MatqΦ = zeros(T,RBVars.Nₛᵖ,RBVars.nₛᵘ,Q)
    @simd for j = 1:RBVars.Nₛᵖ
      Mat_idx = findall(x -> x == j, r_idx)
      MatqΦ[j,:,:] = (MDEIM.Mat[Mat_idx,:]' * RBVars.Φₛ[c_idx[Mat_idx],:])'
    end

    Matₙ = reshape(RBVars.Φₛᵖ' *
      reshape(MatqΦ,RBVars.Nₛᵖ,:),RBVars.nₛᵘ,:,Q)
    RBVars.Bₙ = Matₙ

  else
    assemble_reduced_mat_MDEIM(RBVars.Poisson, MDEIM, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesS{T},
  MDEIM::VMDEIM,
  var::String) where T

  Q = size(MDEIM.Mat)[2]
  Vecₙ = zeros(T,RBVars.nₛᵘ,1,Q)
  @simd for q = 1:Q
    Vecₙ[:,:,q] = RBVars.Φₛ' * Vector{T}(MDEIM.Mat[:, q])
  end
  Vecₙ = reshape(Vecₙ,:,Q)

  if var == "F"
    RBVars.Fₙ = Vecₙ
  elseif var == "H"
    RBVars.Hₙ = Vecₙ
  elseif var == "L"
    RBVars.Lₙ = Vecₙ
  else var == "Lc"
    RBVars.Lcₙ = Vecₙ
  end

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::StokesST{T},
  operators::Vector{String}) where T

  Bₙ = reshape(RBVars.Bₙ, RBVars.nₛᵘ * RBVars.nₛᵖ, :)::Matrix{T}
  affine_vars, affine_names = (Bₙ, RBVars.Lcₙ), ("Bₙ", "Lcₙ")
  affine_entry = get_affine_entries(operators, affine_names)
  save_structures_in_list(affine_vars[affine_entry], affine_names[affine_entry],
    RBInfo.ROM_structures_path)

  MDEIM_vars = (
    RBVars.MDEIM_B.time_idx, RBVars.MDEIM_B.Matᵢ, RBVars.MDEIM_B.idx, RBVars.MDEIM_B.el,
    RBVars.MDEIM_Lc.time_idx, RBVars.MDEIM_Lc.Matᵢ, RBVars.MDEIM_Lc.idx, RBVars.MDEIM_Lc.el)
  MDEIM_names = (
    "time_idx_B","Matᵢ_B","idx_B","el_B",
    "time_idx_Lc","Matᵢ_Lc","idx_Lc","el_Lc")
  save_structures_in_list(MDEIM_vars, MDEIM_names, RBInfo.ROM_structures_path)

  operators_to_pass = setdiff(operators, ("B", "Lc"))
  save_assembled_structures(RBInfo, RBVars.Poisson, operators_to_pass)

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
  FEMSpace::FOMST,
  RBInfo::ROMInfoST,
  RBVars::StokesST{T},
  Param::ParamInfoST,
  var::String) where T

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.α, RBVars.MDEIM_A, "A")::Matrix{T}
  elseif var == "B"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.b, RBVars.MDEIM_M, "B")::Matrix{T}
  elseif var == "M"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.m, RBVars.MDEIM_M, "M")::Matrix{T}
  elseif var == "F"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.f, RBVars.MDEIM_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.h, RBVars.MDEIM_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.MDEIM_L, "L")::Matrix{T}
  elseif var == "Lc"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.MDEIM_Lc, "Lc")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function assemble_param_RHS(
  FEMSpace::FOMST,
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

  RHS_cₙ = RBVars.Φₛ' * (RHS_c * RBVars.Φₜᵘ)
  push!(RBVars.RHSₙ, reshape(RHS_cₙ', :, 1))::Vector{Matrix{T}}

end

function adaptive_loop_on_params(
  FEMSpace::FOMST,
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
    time_err_u[iₜ] = (norm(mean_pointwise_err_u[:,iₜ],RBVars.Xu₀) /
      norm(mean_uₕ_test[:,iₜ],RBVars.Xu₀))
    time_err_p[iₜ] = (norm(mean_pointwise_err_p[:,iₜ],RBVars.Xp₀) /
      norm(mean_pₕ_test[:,iₜ],RBVars.Xp₀))
  end
  for iₛ = 1:RBVars.Nₛᵘ
    space_err_u[iₛ] = norm(mean_pointwise_err_u[iₛ,:])/norm(mean_uₕ_test[iₛ,:])
  end
  for iₛ = 1:RBVars.Nₛᵖ
    space_err_p[iₛ] = norm(mean_pointwise_err_p[iₛ,:])/norm(mean_pₕ_test[iₛ,:])
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

  Φₛ_new = Matrix{T}(qr(Sᵘ[:,ind_t_u]).Q)[:,1:n_adaptive_u[2]]
  Φₜᵘ_new = Matrix{T}(qr(Sᵘ[ind_s_u,:]').Q)[:,1:n_adaptive_u[1]]
  RBVars.nₛᵘ += n_adaptive_u[2]
  RBVars.nₜᵘ += n_adaptive_u[1]
  RBVars.nᵘ = RBVars.nₛᵘ*RBVars.nₜᵘ
  RBVars.Φₛ = Matrix{T}(qr(hcat(RBVars.Φₛ,Φₛ_new)).Q)[:,1:RBVars.nₛᵘ]
  RBVars.Φₜᵘ = Matrix{T}(qr(hcat(RBVars.Φₜᵘ,Φₜᵘ_new)).Q)[:,1:RBVars.nₜᵘ]

  Φₛᵖ_new = Matrix{T}(qr(Sᵖ[:,ind_t_p]).Q)[:,1:n_adaptive_p[2]]
  Φₜᵖ_new = Matrix{T}(qr(Sᵖ[ind_s_p,:]').Q)[:,1:n_adaptive_p[1]]
  RBVars.nₛᵖ += n_adaptive_p[2]
  RBVars.nₜᵖ += n_adaptive_p[1]
  RBVars.nᵖ = RBVars.nₛᵖ*RBVars.nₜᵖ
  RBVars.Φₛᵖ = Matrix{T}(qr(hcat(RBVars.Φₛᵖ,Φₛᵖ_new)).Q)[:,1:RBVars.nₛᵖ]
  RBVars.Φₜᵖ = Matrix{T}(qr(hcat(RBVars.Φₜᵖ,Φₜᵖ_new)).Q)[:,1:RBVars.nₜᵖ]

  RBInfo.save_offline = false
  assemble_offline_structures(RBInfo, RBVars)

  loop_on_params(FEMSpace,RBInfo,RBVars,μ,param_nbs)

end

#################################################################################

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
      RBVars, uₕ_test, RBVars.ũ, RBVars.Xu₀)
    H1_L2_err[i_nb] = H1_L2_err_nb
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_H1_L2_err += H1_L2_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test-RBVars.ũ)/length(param_nbs)

    L2_err_nb, L2_L2_err_nb = compute_errors(
      RBVars, pₕ_test, RBVars.p̃, RBVars.Xp₀)
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

  FEMSpace, μ = get_FEMμ_info(RBInfo)

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
