function get_Aₙ(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Aₙ(RBInfo, RBVars.Poisson)

end

function get_Mₙ(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB)

  get_Mₙ(RBInfo, RBVars.Poisson)

end

function get_Bₙ(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Bₙ(RBInfo, RBVars.Steady)

end

function get_Fₙ(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Fₙ(RBInfo, RBVars.Poisson)

end

function get_Hₙ(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Hₙ(RBInfo, RBVars.Poisson)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::StokesSTGRB{T},
  var::String) where T

  if var == "B"
    println("Assembling affine primal operator B")
    B = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))
    RBVars.Bₙ = zeros(T, RBVars.nₛᵖ, RBVars.nₛᵘ, 1)
    RBVars.Bₙ[:,:,1] = (RBVars.Φₛᵖ)' * B * RBVars.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesSTGRB,
  MDEIM_mat::Matrix,
  row_idx::Vector{Int},
  var::String)

  assemble_reduced_mat_MDEIM(RBVars.Poisson, MDEIM_mat, row_idx, var)

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::StokesSTGRB,
  var::String)

  assemble_affine_vectors(RBInfo, RBVars.Poisson, var)

end

function assemble_reduced_mat_DEIM(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB,
  DEIM_mat::Matrix,
  var::String)

  assemble_reduced_mat_DEIM(RBInfo, RBVars.Poisson, DEIM_mat, var)

end

function assemble_offline_structures(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB,
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
  RBVars::StokesSTGRB)

  if RBInfo.save_offline_structures
    Bₙ = reshape(RBVars.Bₙ, :, 1)
    save_CSV(Bₙ, joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))
  end

end

function get_affine_structures(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  operators = String[]
  append!(operators, get_affine_structures(RBInfo, RBVars.Poisson))
  append!(operators, get_Bₙ(RBInfo, RBVars))

  return operators

end

function get_Q(
  RBInfo::Info,
  RBVars::StokesSTGRB)

  get_Q(RBInfo, RBVars.Poisson)

end

function get_RB_LHS_blocks(
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB{T},
  θᵐ::Matrix,
  θᵃ::Matrix,
  θᵇ::Matrix) where T

  get_RB_LHS_blocks(RBInfo, RBVars.Poisson, θᵐ, θᵃ)

  Φₜᵘᵖ = RBVars.Φₜᵘ' * RBVars.Φₜᵖ
  Bₙᵀ = permutedims(RBVars.Bₙ,[2,1,3])::Array{T,3}
  Bₙᵀ = kron(Bₙᵀ[:,:,1].*θᵇ, Φₜᵘᵖ)::Matrix{T}
  Bₙ = (Bₙᵀ)'::Matrix{T}

  block₂ = -RBInfo.δt*RBInfo.θ * Bₙᵀ
  block₃ = Bₙ

  push!(RBVars.LHSₙ, block₂)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, block₃)::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, zeros(T, RBVars.nᵖ, RBVars.nᵖ))::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBInfo::Info,
  RBVars::StokesSTGRB{T},
  θᶠ::Matrix,
  θʰ::Matrix) where T

  println("Assembling RHS")

  get_RB_RHS_blocks(RBInfo, RBVars.Poisson, θᶠ, θʰ)

  push!(RBVars.RHSₙ, Matrix{T}(undef,0,0))

end

function get_RB_system(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::StokesSTGRB,
  Param::UnsteadyParametricInfo)

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
      if RBInfo.probl_nl["g"]
        build_RB_lifting(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars.Steady,LHS_blocks,RHS_blocks,operators)

end

function build_RB_lifting(
  FEMSpace::UnsteadyProblem,
  RBInfo::ROMInfoUnsteady,
  RBVars::StokesSTGRB{T},
  Param::UnsteadyParametricInfo) where T

  println("Assembling reduced lifting exactly")

  L_t = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")
  L = zeros(T, RBVars.Nₛᵘ+RBVars.Nₛᵖ, RBVars.Nₜ)
  timesθ = get_timesθ(RBInfo)
  for (i,tᵢ) in enumerate(timesθ)
    L[:,i] = L_t(tᵢ)
  end
  Lₙ = Matrix{T}[]
  push!(Lₙ, reshape((vcat(RBVars.Φₛᵘ,RBVars.Φₛᵖ)'*(L*RBVars.Φₜᵘ))',:,1))::Vector{Matrix{T}}
  RBVars.RHSₙ -= Lₙ

end

function build_param_RHS(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::StokesSTGRB,
  Param::UnsteadyParametricInfo)

  build_param_RHS(FEMSpace, RBInfo, RBVars.Poisson, Param)
  push!(RBVars.RHSₙ, zeros(RBVars.nᵖ,1))

end

function get_θ(
  FEMSpace::UnsteadyProblem,
  RBInfo::Info,
  RBVars::StokesSTGRB,
  Param::UnsteadyParametricInfo)

  θᵐ, θᵃ, θᶠ, θʰ  = get_θ(FEMSpace, RBInfo, RBVars.Poisson, Param)
  θᵇ = get_θᵇ(FEMSpace, RBInfo, RBVars, Param)

  return θᵐ, θᵃ, θᵇ, θᶠ, θʰ

end
