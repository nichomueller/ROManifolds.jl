function get_Aₙ(
  RBInfo::Info,
  RBVars::NavierStokesSGRB)

  get_Aₙ(RBInfo, RBVars.Stokes)

end

function get_Bₙ(
  RBInfo::Info,
  RBVars::NavierStokesSGRB)

  get_Bₙ(RBInfo, RBVars.Stokes)

end

function get_Cₙ(
  RBInfo::Info,
  RBVars::NavierStokesSGRB{T}) where T

  if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "Cₙ.csv"))
    println("Importing reduced affine convection matrix")
    Cₙ = load_CSV(Matrix{T}(undef,0,0),
      joinpath(RBInfo.Paths.ROM_structures_path, "Cₙ.csv"))
    RBVars.Cₙ = reshape(Cₙ,RBVars.nₛᵖ,RBVars.nₛᵘ,:)::Array{T,3}
    return [""]
  else
    println("Failed to import Cₙ: must build it")
    return ["C"]
  end

end

function get_Fₙ(
  RBInfo::Info,
  RBVars::NavierStokesSGRB)

  get_Fₙ(RBInfo, RBVars.Stokes)

end

function get_Hₙ(
  RBInfo::Info,
  RBVars::NavierStokesSGRB)

  get_Hₙ(RBInfo, RBVars.Stokes)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::NavierStokesSGRB{T},
  var::String) where T

  assemble_affine_matrices(RBInfo, RBVars.Stokes, var)

end

function assemble_reduced_mat_MDEIM(
  RBVars::NavierStokesSGRB,
  MDEIM_mat::Matrix,
  row_idx::Vector)

  Q = size(MDEIM_mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.Nₛᵘ)
  MatqΦ = zeros(T,RBVars.Nₛᵘ,RBVars.nₛᵘ,Q)
  @simd for j = 1:RBVars.Nₛᵘ
    Mat_idx = findall(x -> x == j, r_idx)
    MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
  end
  Matₙ = reshape(RBVars.Φₛᵘ' *
    reshape(MatqΦ,RBVars.Nₛᵘ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}

  if var == "A"
    RBVars.Aₙ = Matₙ
    RBVars.Qᵃ = Q
  elseif var == "B"
    RBVars.Bₙ = Matₙ
    RBVars.Qᵇ = Q
  elseif var == "C"
    RBVars.Cₙ = Matₙ
    RBVars.Qᶜ = Q
  elseif var == "D"
    RBVars.Dₙ = Matₙ
    RBVars.Qᵈ = Q
  else
    error("Unrecognized variable")
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::NavierStokesSGRB,
  var::String)

  assemble_affine_vectors(RBInfo, RBVars.Stokes, var)

end

function assemble_reduced_mat_DEIM(
  RBVars::NavierStokesSGRB,
  DEIM_mat::Matrix,
  var::String)

  assemble_reduced_mat_DEIM(RBVars.Stokes, DEIM_mat, var)

end

function assemble_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSGRB,
  operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assemble_offline_structures(RBInfo, RBVars.Stokes, operators)

  RBVars.offline_time += @elapsed begin
    if "C" ∈ operators
      assemble_affine_matrices(RBInfo, RBVars, "C")
    end

  end

  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::NavierStokesSGRB)

  if RBInfo.save_offline_structures
    Cₙ = reshape(RBVars.Cₙ, :, RBVars.Qᶜ)::Matrix{T}
    save_CSV(Cₙ, joinpath(RBInfo.Paths.ROM_structures_path, "Cₙ.csv"))
  end

end

function get_affine_structures(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSteady)

  operators = String[]
  append!(operators, get_Cₙ(RBInfo, RBVars))
  append!(operators, get_affine_structures(RBInfo, RBVars.Stokes))

  operators

end

function get_Q(
  RBInfo::Info,
  RBVars::NavierStokesSGRB)

  if RBVars.Qᶜ == 0
    RBVars.Qᶜ = size(RBVars.Cₙ)[end]
  end
  get_Q(RBInfo, RBVars.Stokes)

end

function get_RB_LHS_blocks(
  RBVars::NavierStokesSGRB{T},
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  println("Assembling reduced LHS")

  block₁ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ)
  for q = 1:RBVars.Qᵃ
    block₁ += RBVars.Aₙ[:,:,q] * θᵃ[q]
  end
  for q = 1:RBVars.Qᶜ
    block₁ += RBVars.Cₙ[:,:,q] * θᶜ[q]
  end

  block₂ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ)
  for q = 1:RBVars.Qᵇ
    block₂ += RBVars.Bₙ[:,:,q] * θᵇ[q]
  end

  push!(RBVars.LHSₙ, block₁)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, -block₂')::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, block₂)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ))

end

function get_RB_RHS_blocks(
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSGRB{T},
  θᶠ::Matrix,
  θʰ::Matrix) where T

  get_RB_RHS_blocks(RBInfo, RBVars.Stokes, θᶠ, θʰ)

end

function build_RB_lifting(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSGRB{T},
  Param::SteadyParametricInfo) where T

  println("Assembling reduced lifting exactly")

  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")
  Lₙ = Matrix{T}[]
  push!(Lₙ, vcat(RBVars.Φₛᵘ, RBVars.Φₛᵖ)'*L)
  RBVars.RHSₙ -= Lₙ

end

function build_param_RHS(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::NavierStokesSGRB{T},
  Param::SteadyParametricInfo) where T

  build_param_RHS(FEMSpace, RBInfo, RBVars.Stokes, Param)

end

function get_θ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSGRB{T},
  Param::SteadyParametricInfo) where T

  θᵃ, θᵇ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars.Stokes, Param)
  θᶜ = get_θᶜ(FEMSpace, RBVars, Param)

  return θᵃ, θᵇ, θᶜ, θᶠ, θʰ

end
