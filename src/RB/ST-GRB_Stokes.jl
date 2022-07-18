function get_Aₙ(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Aₙ(RBInfo, RBVars.P)

end

function get_Mₙ(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB)

  get_Mₙ(RBInfo, RBVars.P)

end

function get_Bₙ(
  RBInfo::Info,
  RBVars::StokesSTGRB{T}) where T
  #MODIFY#

  if isfile(joinpath(RBInfo.paths.ROM_structures_path, "Bₙ.csv"))
    println("Importing reduced affine divergence matrix")
    Bₙ = load_CSV(Matrix{T}(undef,0,0),
      joinpath(RBInfo.paths.ROM_structures_path, "Bₙ.csv"))
    RBVars.Bₙ = reshape(Bₙ,RBVars.nₛᵖ,RBVars.nₛᵘ,:)
    return [""]
  else
    println("Failed to import Bₙ: must build it")
    return ["B"]
  end

end

function get_Fₙ(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Fₙ(RBInfo, RBVars.P)

end

function get_Hₙ(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Hₙ(RBInfo, RBVars.P)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::StokesSTGRB{T},
  var::String) where T

  if var == "B"
    println("Assembling affine primal operator B")
    B = load_CSV(sparse([],[],T[]),
      joinpath(RBInfo.paths.FEM_structures_path, "B.csv"))
    RBVars.Bₙ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ, 1)
    RBVars.Bₙ[:,:,1] = (RBVars.Φₛᵖ)' * B * RBVars.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.P, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB,
  MDEIM_mat::Matrix,
  row_idx::Vector{Int64},
  var::String)

  assemble_reduced_mat_MDEIM(RBInfo, RBVars.P, MDEIM_mat, row_idx, var)

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::StokesSTGRB,
  var::String)

  assemble_affine_vectors(RBInfo, RBVars.P, var)

end

function assemble_reduced_mat_DEIM(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB,
  DEIM_mat::Matrix,
  var::String)

  assemble_reduced_mat_DEIM(RBInfo, RBVars.P, DEIM_mat, var)

end

function assemble_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB,
  operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assemble_offline_structures(RBInfo, RBVars.P, operators)

  RBVars.offline_time += @elapsed begin
    if "B" ∈ operators
      assemble_affine_matrices(RBInfo, RBVars, "B")
    end

  end

  save_affine_structures(RBInfo, RBVars)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  if RBInfo.save_offline_structures
    Bₙ = reshape(RBVars.Bₙ, :, 1)
    save_CSV(Bₙ, joinpath(RBInfo.paths.ROM_structures_path, "Bₙ.csv"))
  end

end

function get_affine_structures(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  operators = get_affine_structures(RBInfo, RBVars.P)
  append!(operators, get_Bₙ(RBInfo, RBVars))

  return operators

end

function get_Q(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Q(RBInfo, RBVars.P)

end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB{T},
  θᵐ::Matrix,
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  get_RB_LHS_blocks(RBInfo, RBVars.P, θᵐ, θᵃ)

  Φₜᵘᵖ = RBVars.Φₜᵘ' * RBVars.Φₜᵖ
  Φₜᵘᵖ₁ = RBVars.Φₜᵘ[2:end,:]' * RBVars.Φₜᵖ[1:end-1,:]

  Bₙ = kron(RBVars.Bₙ[:,:,1].*θᵇ, Φₜᵘᵖ')::Matrix{T}
  Bₙᵀ = Bₙ'
  Bₙ₁ᵀ = kron(transpose(RBVars.Bₙ[:,:,1].*θᵇ), Φₜᵘᵖ₁)::Matrix{T}

  block₂ = RBInfo.δt*RBInfo.θ * (RBInfo.θ*Bₙᵀ + (1 - RBInfo.θ)*Bₙ₁ᵀ)
  block₃ = Bₙ

  push!(RBVars.LHSₙ, - block₂)
  push!(RBVars.LHSₙ, block₃)
  push!(RBVars.LHSₙ, Matrix{T}(undef,0,0))

end

function get_RB_RHS_blocks(
  RBInfo::Info,
  RBVars::StokesSTGRB{T},
  θᶠ::Matrix,
  θʰ::Matrix) where T

  println("Assembling RHS")

  get_RB_RHS_blocks(RBInfo, RBVars.P, θᶠ, θʰ)

  push!(RBVars.RHSₙ, Matrix{T}(undef,0,0))

end

function get_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::StokesSTGRB,
  Param::ParametricInfoUnsteady)

  initialize_RB_system(RBVars.S)
  initialize_online_time(RBVars.S)

  LHS_blocks = [1, 2, 3]
  RHS_blocks = [1]

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)

    operators = get_system_blocks(RBInfo,RBVars.S,LHS_blocks,RHS_blocks)

    θᵐ, θᵃ, θᶠ, θʰ, θᵇ  = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBInfo, RBVars, θᵐ, θᵃ, θᵇ)
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        get_RB_RHS_blocks(RBInfo, RBVars, θᶠ, θʰ)
      else
        build_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars.S,LHS_blocks,RHS_blocks,operators)

end

function build_param_RHS(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::StokesSTGRB,
  Param::ParametricInfoUnsteady)

  build_param_RHS(FEMSpace, RBInfo, RBVars.P, Param)
  push!(RBVars.RHSₙ, zeros(RBVars.nᵖ,1))

end

function get_θ(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::StokesSTGRB,
  Param::ParametricInfoUnsteady)

  θᵇ = get_θᵇ(FEMSpace, RBInfo, RBVars, Param)
  #=
  θᵃ = get_θᵃ(FEMSpace, RBInfo, RBVars, Param)
  if !RBInfo.build_parametric_RHS
    θᶠ, θʰ = get_θᶠʰ(FEMSpace, RBInfo, RBVars, Param)
  else
    θᶠ, θʰ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  end =#

  return get_θ(FEMSpace, RBInfo, RBVars.P, Param)..., θᵇ

end
