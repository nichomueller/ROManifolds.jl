################################# OFFLINE ######################################
function POD_space(
  RBInfo::Info,
  RBVars::StokesST)

  POD_space(RBInfo, RBVars.Poisson)

  println("Spatial POD for field p, tolerance: $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars.Steady)
  RBVars.Φₛᵖ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.Xᵖ₀)
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
  MDEIM::MDEIMm,
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
  MDEIM::MDEIMv,
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
  FEMSpace::FEMProblemST,
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

  RHS_cₙ = RBVars.Φₛ' * (RHS_c * RBVars.Φₜᵘ)
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
