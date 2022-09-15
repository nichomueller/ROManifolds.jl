function get_A(
  RBInfo::Info,
  RBVars::StokesSGRB)

  get_A(RBInfo, RBVars.Poisson)

end

function get_B(
  RBInfo::Info,
  RBVars::StokesSGRB{T}) where T

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
  RBVars::StokesSGRB)

  get_F(RBInfo, RBVars.Poisson)

end

function get_H(
  RBInfo::Info,
  RBVars::StokesSGRB)

  get_H(RBInfo, RBVars.Poisson)

end

function get_L(
  RBInfo::Info,
  RBVars::StokesSGRB)

  get_L(RBInfo, RBVars.Poisson)

end

function get_Lc(
  RBInfo::Info,
  RBVars::StokesSGRB)

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
  RBVars::StokesSGRB{T},
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

function assemble_reduced_mat_MDEIM(
  RBVars::StokesSGRB,
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
  RBVars::StokesSGRB,
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

function assemble_reduced_mat_DEIM(
  RBVars::StokesSGRB,
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

function assemble_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSGRB,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.offline_time += @elapsed begin
    for var ∈ intersect(operators, RBInfo.probl_nl)
      if var ∈ ("A", "B")
        assemble_MDEIM_matrices(RBInfo, RBVars, var)
      else
        assemble_DEIM_vectors(RBInfo, RBVars, var)
      end
    end

    for var ∈ setdiff(operators, RBInfo.probl_nl)
      if var ∈ ("A", "B")
        assemble_affine_matrices(RBInfo, RBVars, var)
      else
        assemble_affine_vectors(RBInfo, RBVars, var)
      end
    end
  end

  save_assembled_structures(RBInfo, RBVars)

end

function get_Q(
  RBInfo::Info,
  RBVars::StokesSGRB)

  if RBVars.Qᵇ == 0
    RBVars.Qᵇ = size(RBVars.Bₙ)[end]
  end
  if !RBInfo.build_parametric_RHS
    if RBVars.Qˡᶜ == 0
      RBVars.Qˡᶜ = size(RBVars.Lcₙ)[end]
    end
  end

  get_Q(RBInfo, RBVars.Poisson)

end

function get_RB_LHS_blocks(
  RBVars::StokesSGRB{T},
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  println("Assembling reduced LHS")

  get_RB_LHS_blocks(RBVars.Poisson, θᵃ)

  block₂ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ)
  for q = 1:RBVars.Qᵇ
    block₂ += RBVars.Bₙ[:,:,q] * θᵇ[q]
  end

  push!(RBVars.LHSₙ, -block₂')::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, block₂)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBVars::PoissonSteady{T},
  θᶠ::Matrix,
  θʰ::Matrix,
  θˡ::Matrix,
  θˡᶜ::Matrix) where T

  get_RB_RHS_blocks(RBVars.Poisson, θᶠ, θʰ, θˡ)
  push!(RBVars.RHSₙ, -RBVars.Lcₙ * θˡᶜ)::Vector{Matrix{T}}

end

function build_param_RHS(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSGRB{T},
  Param::SteadyParametricInfo) where T

  build_param_RHS(FEMSpace, RBInfo, RBVars.Poisson, Param)

  Lc = assemble_FEM_structure(FEMSpace, RBInfo, Param, "Lc")
  push!(RBVars.RHSₙ, reshape(-RBVars.Φₛᵖ' * L, :, 1)::Matrix{T})

end
