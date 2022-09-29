################################# OFFLINE ######################################

function check_norm_matrix(RBVars::StokesS)
  isempty(RBVars.Xᵘ₀) || isempty(RBVars.Xᵖ₀)
end

function PODs_space(
  RBInfo::Info,
  RBVars::StokesS)

  PODs_space(RBInfo,RBVars.Poisson)

  println("Performing the spatial POD for field p, using a tolerance of $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars)
  RBVars.Φₛᵖ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.Xᵖ₀)
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function assemble_constraint_matrix(
  RBInfo::Info,
  RBVars::StokesS{T}) where T

  FEMSpace, μ = get_FEMProblem_info(RBInfo.FEMInfo)

  constraint_matrix = zeros(T, RBVars.Nₛᵘ, RBVars.nₛ)

  for k = 1:RBVars.nₛ
    println("Column number $k, constraint matrix")
    Param = get_ParamInfo(RBInfo, μ[k])
    B_k = Matrix{T}(assemble_FEM_structure(FEMSpace, RBInfo, Param, "B")')
    constraint_matrix[:, k] = B_k * RBVars.Sᵖ[:, k]
  end

  constraint_matrix

end

function primal_supremizers(
  RBInfo::Info,
  RBVars::StokesS{T}) where T

  println("Computing primal supremizers")

  if "B" ∈ RBInfo.probl_nl
    println("Matrix Bᵀ is nonaffine: must assemble constraint_matrix ∀ μ")
    constraint_matrix = assemble_constraint_matrix(RBInfo, RBVars)
  else
    println("Loading matrix Bᵀ")
    constraint_matrix = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))'
  end

  supr_primal = solve_cholesky(RBVars.Xᵘ₀, constraint_matrix * RBVars.Φₛᵖ)

  min_norm = 1e16
  for i = 1:size(supr_primal)[2]

    println("Normalizing primal supremizer $i")

    for j in 1:RBVars.nₛᵘ
      supr_primal[:, i] -= mydot(supr_primal[:, i], RBVars.Φₛᵘ[:, j], RBVars.Xᵘ₀) /
      mynorm(RBVars.Φₛᵘ[:, j], RBVars.Xᵘ₀) * RBVars.Φₛᵘ[:, j]
    end
    for j in 1:i
      supr_primal[:, i] -= mydot(supr_primal[:, i], supr_primal[:, j], RBVars.Xᵘ₀) /
      mynorm(supr_primal[:, j], RBVars.Xᵘ₀) * supr_primal[:, j]
    end

    supr_norm = mynorm(supr_primal[:, i], RBVars.Xᵘ₀)
    min_norm = min(supr_norm, min_norm)
    println("Norm supremizers: $supr_norm")
    supr_primal[:, i] /= supr_norm

  end

  println("Primal supremizers enrichment ended with norm: $min_norm")

  supr_primal

end

function supr_enrichment_space(
  RBInfo::Info,
  RBVars::StokesS)

  supr_primal = primal_supremizers(RBInfo, RBVars)
  RBVars.Φₛᵘ = hcat(RBVars.Φₛᵘ, supr_primal)
  RBVars.nₛᵘ = size(RBVars.Φₛᵘ)[2]

end

function get_generalized_coordinates(
  RBInfo::ROMInfoS,
  RBVars::StokesS,
  snaps=nothing)

  get_norm_matrix(RBInfo, RBVars)
  if isnothing(snaps) || maximum(snaps) > RBInfo.nₛ
    snaps = 1:RBInfo.nₛ
  end

  get_generalized_coordinates(RBInfo, RBVars.Poisson, snaps)

  Φₛᵖ_normed = RBVars.Xᵖ₀*RBVars.Φₛᵖ
  RBVars.p̂ = RBVars.Sᵖ[:,snaps]*Φₛᵖ_normed
  if RBInfo.save_offline_structures
    save_CSV(RBVars.p̂, joinpath(RBInfo.ROM_structures_path, "p̂.csv"))
  end

end

function set_operators(
  RBInfo::Info,
  RBVars::StokesS)

  operators = vcat(["B"], set_operators(RBInfo, RBVars.Poisson))
  if !RBInfo.online_RHS
    append!(operators, ["Lc"])
  end

  operators

end

function get_A(
  RBInfo::Info,
  RBVars::StokesS)

  get_A(RBInfo, RBVars.Poisson)

end

function get_B(
  RBInfo::Info,
  RBVars::StokesS{T}) where T

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
  RBVars::StokesS)

  get_F(RBInfo, RBVars.Poisson)

end

function get_H(
  RBInfo::Info,
  RBVars::StokesS)

  get_H(RBInfo, RBVars.Poisson)

end

function get_L(
  RBInfo::Info,
  RBVars::StokesS)

  get_L(RBInfo, RBVars.Poisson)

end

function get_Lc(
  RBInfo::Info,
  RBVars::StokesS{T}) where T

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
  RBVars::StokesS{T},
  var::String) where T

  if var == "B"
    println("Assembling affine reduced B")
    B = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))
    RBVars.Bₙ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ, 1)
    RBVars.Bₙ[:,:,1] = (RBVars.Φₛᵖ)' * B * RBVars.Φₛᵘ
    RBVars.Qᵇ = 1
  if var == "Lc"
    println("Assembling affine reduced Lc")
    Lc = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_structures_path(RBInfo), "Lc.csv"))
    RBVars.Lcₙ = RBVars.Φₛᵖ' * Lc
    RBVars.Qˡᶜ = 1
  else
    assemble_affine_vectors(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_MDEIM_structures(
  RBInfo::ROMInfoS,
  RBVars::StokesS,
  var::String)

  println("The matrix $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "A"
    if isempty(RBVars.MDEIM_A.Mat)
      MDEIM_offline!(RBVars.MDEIM_A, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_A, var)
  elseif var == "B"
    if isempty(RBVars.MDEIM_B.Mat)
      MDEIM_offline!(RBVars.MDEIM_B, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_B, var)
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
  RBVars::StokesS{T},
  MDEIM::MDEIMmS,
  var::String) where T

  if var == "B"
    Q = size(MDEIM.Mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, RBVars.Nₛᵖ)
    MatqΦ = zeros(T,RBVars.Nₛᵖ,RBVars.nₛᵘ,Q)::Array{T,3}
    @simd for j = 1:RBVars.Nₛᵖ
      Mat_idx = findall(x -> x == j, r_idx)
      MatqΦ[j,:,:] = (MDEIM.Mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
    end

    Matₙ = reshape(RBVars.Φₛᵖ' *
      reshape(MatqΦ,RBVars.Nₛᵖ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}
    RBVars.Bₙ = Matₙ
    RBVars.Qᵇ = Q

  else
    assemble_reduced_mat_MDEIM(RBVars.Poisson, MDEIM, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesS{T},
  MDEIM::MDEIMvS,
  var::String) where T

  if var == "Lc"
    Q = size(MDEIM.Mat)[2]
    Vecₙ = zeros(T, RBVars.nₛᵖ, 1, Q)
    @simd for q = 1:Q
      Vecₙ[:,:,q] = RBVars.Φₛᵖ' * Vector{T}(MDEIM.Mat[:, q])
    end
    RBVars.Lcₙ = reshape(Vecₙ, :, Q)
    RBVars.Qˡᶜ = Q
  else
    assemble_reduced_mat_DEIM(RBVars.Poisson, MDEIM, var)
  end

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::StokesS{T},
  operators::Vector{String}) where T

  Bₙ = reshape(RBVars.Bₙ, RBVars.nₛᵘ * RBVars.nₛᵖ, :)::Matrix{T}
  affine_vars, affine_names = (Bₙ, RBVars.Lcₙ), ("Bₙ", "Lcₙ")
  affine_entry = get_affine_entries(operators, affine_names)
  save_structures_in_list(affine_vars[affine_entry], affine_names[affine_entry],
    RBInfo.ROM_structures_path)

  M_DEIM_vars = (
    RBVars.MDEIM_B.Matᵢ, RBVars.MDEIM_B.idx, RBVars.MDEIM_B.el,
    RBVars.MDEIM_Lc.Matᵢ, RBVars.MDEIM_Lc.idx, RBVars.MDEIM_Lc.el)
  M_DEIM_names = (
    "Matᵢ_A","idx_A","el_A",
    "Matᵢ_Lc","idx_Lc","el_Lc")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path)

  operators_to_pass = setdiff(operators, ("B", "Lc"))
  save_assembled_structures(RBInfo, RBVars.Poisson, operators_to_pass)

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::Info,
  RBVars::StokesS,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int})

  get_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::StokesS,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  if "A" ∉ RBInfo.probl_nl && "B" ∉ RBInfo.probl_nl && "LHS" ∈ operators
    for i = LHS_blocks
      LHSₙi = "LHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.LHSₙ[i],joinpath(RBInfo.ROM_structures_path, LHSₙi))
    end
  end
  if ("F" ∉ RBInfo.probl_nl && "H" ∉ RBInfo.probl_nl && "L" ∉ RBInfo.probl_nl
      && "Lc" ∉ RBInfo.probl_nl && "RHS" ∈ operators)
    for i = RHS_blocks
      RHSₙi = "RHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.RHSₙ[i],joinpath(RBInfo.ROM_structures_path, RHSₙi))
    end
  end

end

function get_θ_matrix(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::StokesS,
  Param::ParamInfoS,
  var::String) where T

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.α, RBVars.MDEIM_A, "A")::Matrix{T}
  elseif var == "B"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.b, RBVars.MDEIM_B, "B")::Matrix{T}
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

function get_Q(RBVars::StokesS)

  RBVars.Qᵇ = size(RBVars.Bₙ)[end]
  RBVars.Qˡᶜ = size(RBVars.Lcₙ)[end]

  get_Q(RBVars.Poisson)

end

function assemble_param_RHS(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::StokesS{T},
  Param::ParamInfoS) where T

  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")
  push!(RBVars.RHSₙ, reshape(RBVars.Φₛᵘ' * (F + H - L), :, 1)::Matrix{T})

  Lc = assemble_FEM_structure(FEMSpace, RBInfo, Param, "Lc")
  push!(RBVars.RHSₙ, reshape(- RBVars.Φₛᵖ' * Lc, :, 1)::Matrix{T})

end
