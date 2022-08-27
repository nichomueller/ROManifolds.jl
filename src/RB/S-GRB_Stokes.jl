function get_Aₙ(
  RBInfo::Info,
  RBVars::StokesSGRB)

  get_Aₙ(RBInfo, RBVars.Poisson)

end

function get_Bₙ(
  RBInfo::Info,
  RBVars::StokesSGRB{T}) where T

  if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "Bₙ.csv"))
    println("Importing reduced affine divergence matrix")
    Bₙ = load_CSV(Matrix{T}(undef,0,0),
      joinpath(RBInfo.Paths.ROM_structures_path, "Bₙ.csv"))
    RBVars.Bₙ = reshape(Bₙ,RBVars.nₛᵖ,RBVars.nₛᵘ,:)::Array{T,3}
    RBVars.Qᵇ = size(RBVars.Bₙ)[3]
    return [""]
  else
    println("Failed to import Bₙ: must build it")
    return ["B"]
  end

end

function get_Fₙ(
  RBInfo::Info,
  RBVars::StokesSGRB)

  get_Fₙ(RBInfo, RBVars.Poisson)

end

function get_Hₙ(
  RBInfo::Info,
  RBVars::StokesSGRB)

  get_Hₙ(RBInfo, RBVars.Poisson)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::StokesSGRB{T},
  var::String) where T

  if var == "B"
    println("Assembling affine reduced B")
    B = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))
    RBVars.Bₙ = reshape((RBVars.Φₛᵖ)'*B*RBVars.Φₛᵘ,RBVars.nₛᵖ,RBVars.nₛᵘ,1)
  else
    assemble_affine_matrices(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesSGRB,
  MDEIM_mat::Matrix,
  row_idx::Vector)

  assemble_reduced_mat_MDEIM(RBVars.Poisson, MDEIM_mat, row_idx)

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::StokesSGRB,
  var::String)

  assemble_affine_vectors(RBInfo, RBVars.Poisson, var)

end

function assemble_reduced_mat_DEIM(
  RBVars::StokesSGRB,
  DEIM_mat::Matrix,
  var::String)

  assemble_reduced_mat_DEIM(RBVars.Poisson, DEIM_mat, var)

end

function assemble_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSGRB,
  operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assemble_offline_structures(RBInfo, RBVars.Poisson, operators)

  RBVars.offline_time += @elapsed begin
    if "B" ∈ operators
      assemble_affine_matrices(RBInfo, RBVars, "B")
    end

  end

  save_affine_structures(RBInfo, RBVars)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::StokesSGRB)

  if RBInfo.save_offline_structures
    Bₙ = reshape(RBVars.Bₙ, :, 1)
    save_CSV(Bₙ, joinpath(RBInfo.Paths.ROM_structures_path, "Bₙ.csv"))
  end

end

function get_affine_structures(
  RBInfo::ROMInfoSteady,
  RBVars::StokesSteady)

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars.Poisson))
  append!(operators, get_Bₙ(RBInfo, RBVars))

  operators

end

function get_Q(
  RBInfo::Info,
  RBVars::StokesSGRB)

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
  push!(RBVars.LHSₙ, zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ))::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBVars::PoissonSteady{T},
  θᶠ::Array,
  θʰ::Array) where T

  get_RB_RHS_blocks(RBVars.Poisson, θᶠ, θʰ)
  push!(RBVars.RHSₙ, zeros(T, RBVars.nₛᵖ, 1))::Vector{Matrix{T}}

end

function build_param_RHS(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSGRB{T},
  Param::SteadyParametricInfo) where T

  build_param_RHS(FEMSpace, RBInfo, RBVars.Poisson, Param)
  push!(RBVars.RHSₙ, zeros(T, RBVars.nₛᵖ, 1))

end

function get_θ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSGRB{T},
  Param::SteadyParametricInfo) where T

  θᵃ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars.Poisson, Param)
  θᵇ = get_θᵇ(FEMSpace, RBInfo, RBVars, Param)

  return θᵃ, θᵇ, θᶠ, θʰ

end
