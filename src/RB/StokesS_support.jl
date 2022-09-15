################################# OFFLINE ######################################

function check_norm_matrix(RBVars::StokesSteady)
  isempty(RBVars.Xᵘ₀) || isempty(RBVars.Xᵖ₀)
end

function PODs_space(
  RBInfo::Info,
  RBVars::StokesSteady)

  PODs_space(RBInfo,RBVars.Poisson)

  println("Performing the spatial POD for field p, using a tolerance of $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars)
  RBVars.Φₛᵖ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.Xᵖ₀)
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function primal_supremizers(
  RBInfo::Info,
  RBVars::StokesSteady{T}) where T

  println("Computing primal supremizers")

  constraint_mat = load_CSV(sparse([],[],T[]),
    joinpath(get_FEM_structures_path(RBInfo), "B.csv"))'

  supr_primal = Matrix{T}(RBVars.Xᵘ₀) \ (Matrix{T}(constraint_mat) * RBVars.Φₛᵖ)

  min_norm = 1e16
  for i = 1:size(supr_primal)[2]

    println("Normalizing primal supremizer $i")

    for j in 1:RBVars.nₛᵘ
      supr_primal[:, i] -= mydot(supr_primal[:, i], RBVars.Φₛᵘ[:,j], RBVars.Xᵘ₀) /
      mynorm(RBVars.Φₛᵘ[:,j], RBVars.Xᵘ₀) * RBVars.Φₛᵘ[:,j]
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
  RBVars::StokesSteady)

  supr_primal = primal_supremizers(RBInfo, RBVars)
  RBVars.Φₛᵘ = hcat(RBVars.Φₛᵘ, supr_primal)
  RBVars.nₛᵘ = size(RBVars.Φₛᵘ)[2]

end

function get_generalized_coordinates(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
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
  RBVars::StokesSteady)

  append!(["B", "Lc"], set_operators(RBInfo, RBVars.Poisson))

end

function get_A(
  RBInfo::Info,
  RBVars::StokesSteady)

  get_A(RBInfo, RBVars.Poisson)

end

function get_B(
  RBInfo::Info,
  RBVars::StokesSteady{T}) where T

  if "B" ∈ RBInfo.probl_nl

    if isfile(joinpath(RBInfo.ROM_structures_path, "MDEIMᵢ_B.csv"))
      println("Importing MDEIM offline structures, B")
      (RBVars.MDEIMᵢ_B, RBVars.MDEIM_idx_B, RBVars.row_idx_B, RBVars.sparse_el_B) =
        load_structures_in_list(("MDEIMᵢ_B", "MDEIM_idx_B", "row_idx_B", "sparse_el_B"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)
      return [""]
    else
      println("Failed to import MDEIM offline structures for
        B: must build them")
      return ["B"]
    end

  else

    if isfile(joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))
      println("Importing reduced affine divergence matrix")
      Bₙ = load_CSV(Matrix{T}(undef,0,0),
        joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))
      RBVars.Bₙ = reshape(Bₙ,RBVars.nₛᵖ,RBVars.nₛᵘ,:)::Array{T,3}
      RBVars.Qᵇ = size(RBVars.Bₙ)[3]
      return [""]
    else
      println("Failed to import Bₙ: must build it")
      return ["B"]
    end

  end

end

function get_F(
  RBInfo::Info,
  RBVars::StokesSteady)

  get_F(RBInfo, RBVars.Poisson)

end

function get_H(
  RBInfo::Info,
  RBVars::StokesSteady)

  get_H(RBInfo, RBVars.Poisson)

end

function get_L(
  RBInfo::Info,
  RBVars::StokesSteady)

  get_L(RBInfo, RBVars.Poisson)

end

function get_Lc(
  RBInfo::Info,
  RBVars::StokesSteady)

  if "Lc" ∈ RBInfo.probl_nl

    if isfile(joinpath(RBInfo.ROM_structures_path, "DEIMᵢ_Lc.csv"))
      println("Importing DEIM offline structures, L")
      (RBVars.DEIMᵢ_Lc, RBVars.DEIM_idx_Lc, RBVars.sparse_el_Lc) =
        load_structures_in_list(("DEIMᵢ_Lc", "DEIM_idx_Lc", "sparse_el_Lc"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)
      return [""]
    else
      println("Failed to import DEIM offline structures for Lc: must build them")
      return ["Lc"]
    end

  else

    if isfile(joinpath(RBInfo.ROM_structures_path, "Lcₙ.csv"))
      println("Importing Lcₙ")
      RBVars.Lcₙ = load_CSV(Matrix{T}(undef,0,0),
        joinpath(RBInfo.ROM_structures_path, "Lcₙ.csv"))
      return [""]
    else
      println("Failed to import Lcₙ: must build it")
      return ["Lc"]
    end

  end

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::StokesSteady{T},
  var::String) where T

  if var == "B"
    println("Assembling affine reduced B")
    RBVars.Qᵇ = 1
    B = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))
    RBVars.Bₙ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ, 1)
    RBVars.Bₙ[:,:,1] = (RBVars.Φₛᵖ)' * B * RBVars.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
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
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesSteady,
  MDEIM_mat::Matrix,
  row_idx::Vector,
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
  RBVars::StokesSteady,
  var::String)

  if var == "Lc"
    RBVars.Qˡᶜ = 1
    println("Assembling affine reduced lifting term, continuity")
    Lc = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_structures_path(RBInfo), "Lc.csv"))
    RBVars.Lcₙ = RBVars.Φₛᵖ' * Lc
  else
    assemble_affine_vectors(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  var::String)

  println("The vector $var is non-affine:
    running the DEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "F"
    if isempty(RBVars.DEIM_mat_F)
      RBVars.DEIM_mat_F, RBVars.DEIM_idx_F, RBVars.DEIMᵢ_F, RBVars.sparse_el_F =
        DEIM_offline(RBInfo,"F")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_F,"F")
  elseif var == "H"
    if isempty(RBVars.DEIM_mat_H)
      RBVars.DEIM_mat_H, RBVars.DEIM_idx_H, RBVars.DEIMᵢ_H, RBVars.sparse_el_H =
        DEIM_offline(RBInfo,"H")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_H,"H")
  elseif var == "L"
    if isempty(RBVars.DEIM_mat_L)
      RBVars.DEIM_mat_L, RBVars.DEIM_idx_L, RBVars.DEIMᵢ_L, RBVars.sparse_el_L =
        DEIM_offline(RBInfo,"L")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_L,"L")
  elseif var == "Lc"
    if isempty(RBVars.DEIM_mat_Lc)
      RBVars.DEIM_mat_Lc, RBVars.DEIM_idx_Lc, RBVars.DEIMᵢ_Lc, RBVars.sparse_el_Lc =
        DEIM_offline(RBInfo,"Lc")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_Lc,"Lc")
  else
    error("Unrecognized variable on which to perform DEIM")
  end

end

function assemble_reduced_mat_DEIM(
  RBVars::StokesSteady,
  DEIM_mat::Matrix,
  var::String)

  if var == "Lc"
    Q = size(DEIM_mat)[2]
    Vecₙ = zeros(T, RBVars.nₛᵖ, 1, Q)
    @simd for q = 1:Q
      Vecₙ[:,:,q] = RBVars.Φₛᵖ' * Vector{T}(DEIM_mat[:, q])
    end
    RBVars.Lcₙ = reshape(Vecₙ, :, Q)
    RBVars.Qˡᶜ = Q
  else
    assemble_reduced_mat_DEIM(RBVars.Poisson, DEIM_mat, var)
  end

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::PoissonSteady)

  affine_vars = (reshape(RBVars.Bₙ, :, RBVars.Qᵇ)::Matrix{T}, RBVars.Lcₙ)
  affine_names = ("Bₙ", "Lcₙ")
  save_structures_in_list(affine_vars, affine_names, RBInfo.ROM_structures_path)

  M_DEIM_vars = (
    RBVars.MDEIM_mat_B, RBVars.MDEIMᵢ_B, RBVars.MDEIM_idx_B, RBVars.row_idx_B,
    RBVars.sparse_el_B, RBVars.DEIM_mat_Lc, RBVars.DEIMᵢ_Lc, RBVars.DEIM_idx_Lc,)
    RBVars.sparse_el_Lc
  M_DEIM_names = (
    "MDEIM_mat_B","MDEIMᵢ_B","MDEIM_idx_B","row_idx_B","sparse_el_B",
    "DEIM_mat_Lc","DEIMᵢ_Lc","DEIM_idx_Lc","sparse_el_Lc")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path)

  save_assembled_structures(RBInfo, RBVars.Poisson)

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::Info,
  RBVars::StokesSteady,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int})

  get_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::StokesSteady,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  save_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks, operators)

end

function get_θ_matrix(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  Param::SteadyParametricInfo,
  var::String)

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param.α, RBVars.MDEIMᵢ_A,
      RBVars.MDEIM_idx_A, RBVars.sparse_el_A, "A")::Matrix{T}
  elseif var == "B"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param.b, RBVars.MDEIMᵢ_B,
      RBVars.MDEIM_idx_B, RBVars.sparse_el_B, "B")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_θ_vector(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady,
  Param::SteadyParametricInfo,
  var::String)

  if var == "F"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.f, RBVars.DEIMᵢ_F,
      RBVars.DEIM_idx_F, RBVars.sparse_el_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.h, RBVars.DEIMᵢ_H,
      RBVars.DEIM_idx_H, RBVars.sparse_el_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.g, RBVars.DEIMᵢ_L,
      RBVars.DEIM_idx_L, RBVars.sparse_el_L, "L")::Matrix{T}
  elseif var == "Lc"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.g, RBVars.DEIMᵢ_Lc,
      RBVars.DEIM_idx_Lc, RBVars.sparse_el_Lc, "Lc")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_Q(
  RBInfo::Info,
  RBVars::StokesSteady)

  if RBVars.Qᵇ == 0
    RBVars.Qᵇ = size(RBVars.Bₙ)[end]
  end
  if !RBInfo.assemble_parametric_RHS
    if RBVars.Qˡᶜ == 0
      RBVars.Qˡᶜ = size(RBVars.Lcₙ)[end]
    end
  end

  get_Q(RBInfo, RBVars.Poisson)

end

function assemble_param_RHS(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady{T},
  Param::SteadyParametricInfo) where T

  assemble_param_RHS(FEMSpace, RBInfo, RBVars.Poisson, Param)

  Lc = assemble_FEM_structure(FEMSpace, RBInfo, Param, "Lc")
  push!(RBVars.RHSₙ, reshape(-RBVars.Φₛᵖ' * L, :, 1)::Matrix{T})

end
