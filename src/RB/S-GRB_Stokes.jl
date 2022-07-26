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
    RBVars.Bₙ = reshape(Bₙ,RBVars.nₛᵖ,RBVars.nₛᵘ,:)
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
    assemble_affine_matrices(RBInfo, RBVars, var)
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

  operators = get_affine_structures(RBInfo, RBVars.Poisson)
  append!(operators, get_Bₙ(RBInfo, RBVars))

  operators

end

function get_Q(
  RBInfo::Info,
  RBVars::StokesSGRB)

  get_Q(RBInfo, RBVars.Poisson)

end

function get_RB_LHS_blocks(
  ::ROMInfoSteady,
  RBVars::StokesSGRB{T},
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  Aₙ_μ =  assemble_parametric_structure(θᵃ, RBVars.Aₙ)
  Bₙ_μ =  assemble_parametric_structure(θᵇ, RBVars.Bₙ)

  push!(RBVars.LHSₙ, Aₙ_μ)
  push!(RBVars.LHSₙ, -Bₙ_μ')
  push!(RBVars.LHSₙ, Bₙ_μ)
  push!(RBVars.LHSₙ, Matrix{T}(undef,0,0))

end

function get_RB_RHS_blocks(
  ::ROMInfoSteady,
  RBVars::StokesSGRB{T},
  θᶠ::Matrix,
  θʰ::Matrix) where T

  Fₙ_μ = assemble_parametric_structure(θᶠ, RBVars.Fₙ)
  Hₙ_μ = assemble_parametric_structure(θʰ, RBVars.Hₙ)

  push!(RBVars.RHSₙ, Fₙ_μ + Hₙ_μ)
  push!(RBVars.RHSₙ, Matrix{T}(undef,0,0))

end

function build_param_RHS(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSGRB{T},
  Param::ParametricInfoSteady,
  ::Array) where T

  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  Fₙ_μ = reshape((RBVars.Φₛᵘ)'*F,:,1)::Matrix{T}
  Hₙ_μ = reshape((RBVars.Φₛᵘ)'*H,:,1)::Matrix{T}

  push!(RBVars.RHSₙ, Fₙ_μ + Hₙ_μ)
  push!(RBVars.RHSₙ, zeros(RBVars.nₛᵖ,1))

end

function get_θ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::StokesSGRB{T},
  Param::ParametricInfoSteady) where T

  θᵇ = get_θᵇ(FEMSpace, RBInfo, RBVars, Param)
  #=
  θᵃ = get_θᵃ(FEMSpace, RBInfo, RBVars, Param)
  if !RBInfo.build_parametric_RHS
    θᶠ, θʰ = get_θᶠʰ(FEMSpace, RBInfo, RBVars, Param)
  else
    θᶠ, θʰ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  end =#

  return get_θ(FEMSpace, RBInfo, RBVars.Poisson, Param)..., θᵇ

end
