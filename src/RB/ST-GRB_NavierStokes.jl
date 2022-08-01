function get_Aₙ(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  get_Aₙ(RBInfo, RBVars.Stokes)

end

function get_Mₙ(
  RBInfo::ROMInfoUnsteady,
  RBVars::NavierStokesSTGRB)

  get_Mₙ(RBInfo, RBVars.Stokes)

end

function get_Bₙ(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  get_Bₙ(RBInfo, RBVars.Stokes)

end

function get_Fₙ(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  get_Fₙ(RBInfo, RBVars.Stokes)

end

function get_Hₙ(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  get_Hₙ(RBInfo, RBVars.Stokes)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  var::String)

  assemble_affine_matrices(RBInfo, RBVars.Stokes, var)

end

function assemble_reduced_mat_MDEIM(
  RBInfo::ROMInfoUnsteady,
  RBVars::NavierStokesSTGRB,
  MDEIM_mat::Matrix,
  row_idx::Vector{Int},
  var::String)

  assemble_reduced_mat_MDEIM(RBInfo, RBVars.Stokes, MDEIM_mat, row_idx, var)

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  var::String)

  assemble_affine_vectors(RBInfo, RBVars.Stokes, var)

end

function assemble_reduced_mat_DEIM(
  RBInfo::ROMInfoUnsteady,
  RBVars::NavierStokesSTGRB,
  DEIM_mat::Matrix,
  var::String)

  assemble_reduced_mat_DEIM(RBInfo, RBVars.Stokes, DEIM_mat, var)

end

function assemble_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::NavierStokesSTGRB,
  operators=nothing)

  if isnothing(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assemble_offline_structures(RBInfo, RBVars.Stokes, operators)

  RBVars.offline_time += @elapsed begin
    if "B" ∈ operators
      assemble_affine_matrices(RBInfo, RBVars, "B")
    end

  end

  save_affine_structures(RBInfo, RBVars)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  if RBInfo.save_offline_structures
    Bₙ = reshape(RBVars.Bₙ, :, 1)
    save_CSV(Bₙ, joinpath(RBInfo.Paths.ROM_structures_path, "Bₙ.csv"))
  end

end

function get_affine_structures(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  operators = get_affine_structures(RBInfo, RBVars.Stokes)
  append!(operators, get_Bₙ(RBInfo, RBVars))

  return operators

end

function get_Q(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB)

  get_Q(RBInfo, RBVars.Stokes)

end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::NavierStokesSTGRB{T},
  θᵐ::Matrix,
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  get_RB_LHS_blocks(RBInfo, RBVars.Stokes, θᵐ, θᵃ)

  Φₜᵘᵖ = RBVars.Φₜᵘ' * RBVars.Φₜᵖ
  Bₙᵀ = permutedims(RBVars.Bₙ,[2,1,3])::Array{T,3}
  Bₙᵀ = kron(Bₙᵀ[:,:,1].*θᵇ, Φₜᵘᵖ)::Matrix{T}
  Bₙ = (Bₙᵀ)'::Matrix{T}

  block₂ = -RBInfo.δt*RBInfo.θ * Bₙᵀ
  block₃ = Bₙ

  push!(RBVars.LHSₙ, block₂)
  push!(RBVars.LHSₙ, block₃)
  push!(RBVars.LHSₙ, zeros(T, RBVars.nᵖ, RBVars.nᵖ))

end

function get_RB_RHS_blocks(
  RBInfo::Info,
  RBVars::NavierStokesSTGRB{T},
  θᶠ::Matrix,
  θʰ::Matrix) where T

  println("Assembling RHS")

  get_RB_RHS_blocks(RBInfo, RBVars.Stokes, θᶠ, θʰ)

  push!(RBVars.RHSₙ, Matrix{T}(undef,0,0))

end

function get_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  Param::ParametricInfoUnsteady)

  initialize_RB_system(RBVars.Steady)
  initialize_online_time(RBVars.Steady)

  LHS_blocks = [1, 2, 3]
  RHS_blocks = [1]

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)

    operators = get_system_blocks(RBInfo,RBVars.Steady,LHS_blocks,RHS_blocks)

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

  save_system_blocks(RBInfo,RBVars.Steady,LHS_blocks,RHS_blocks,operators)

end

function build_param_RHS(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  Param::ParametricInfoUnsteady)

  build_param_RHS(FEMSpace, RBInfo, RBVars.Stokes, Param)
  push!(RBVars.RHSₙ, zeros(RBVars.nᵖ,1))

end

function get_θ(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::NavierStokesSTGRB,
  Param::ParametricInfoUnsteady)

  θᵐ, θᵃ, θᶠ, θʰ  = get_θ(FEMSpace, RBInfo, RBVars.Stokes, Param)
  θᵇ = get_θᵇ(FEMSpace, RBInfo, RBVars, Param)

  return θᵐ, θᵃ, θᵇ, θᶠ, θʰ

end
