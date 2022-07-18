function get_Aₙ(
  RBInfo::Info,
  RBVars::PoissonSGRB{T}) where T

  if isfile(joinpath(RBInfo.Paths.ROM_structures_path, "Aₙ.csv"))
    println("Importing reduced affine stiffness matrix")
    Aₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.ROM_structures_path, "Aₙ.csv"))
    RBVars.Aₙ = reshape(Aₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}
    RBVars.Qᵃ = size(RBVars.Aₙ)[3]
    return [""]
  else
    println("Failed to import Aₙ: must build it")
    return ["A"]
  end

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::PoissonSGRB{T},
  var::String) where T

  if var == "A"
    RBVars.Qᵃ = 1
    println("Assembling affine reduced stiffness")
    A = load_CSV(sparse([],[],T[]), joinpath(RBInfo.Paths.FEM_structures_path, "A.csv"))
    RBVars.Aₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
    RBVars.Aₙ[:,:,1] = (RBVars.Φₛᵘ)' * A * RBVars.Φₛᵘ
  else
    error("Unrecognized variable to load")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonSGRB{T},
  MDEIM_mat::Matrix,
  row_idx::Vector) where T

  Q = size(MDEIM_mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.Nₛᵘ)
  MatqΦ = zeros(T,RBVars.Nₛᵘ,RBVars.nₛᵘ,Q)
  @simd for j = 1:RBVars.Nₛᵘ
    Mat_idx = findall(x -> x == j, r_idx)
    MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
  end
  RBVars.Aₙ = reshape(RBVars.Φₛᵘ' *
    reshape(MatqΦ,RBVars.Nₛᵘ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}
  RBVars.Qᵃ = Q

end

function assemble_reduced_mat_DEIM(
  RBVars::PoissonSGRB{T},
  DEIM_mat::Matrix,
  var::String) where T

  Q = size(DEIM_mat)[2]
  Vecₙ = zeros(T,RBVars.nₛᵘ,1,Q)
  @simd for q = 1:Q
    Vecₙ[:,:,q] = RBVars.Φₛᵘ' * Vector{T}(DEIM_mat[:, q])
  end
  Vecₙ = reshape(Vecₙ,:,Q)

  if var == "F"
    RBVars.Fₙ = Vecₙ
    RBVars.Qᶠ = Q
  else var == "H"
    RBVars.Hₙ = Vecₙ
    RBVars.Qʰ = Q
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::PoissonSGRB{T},
  var::String) where T

  if var == "F"
    RBVars.Qᶠ = 1
    println("Assembling affine reduced forcing term")
    F = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.FEM_structures_path, "F.csv"))
    RBVars.Fₙ = (RBVars.Φₛᵘ)' * F
  elseif var == "H"
    RBVars.Qʰ = 1
    println("Assembling affine reduced Neumann term")
    H = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.Paths.FEM_structures_path, "H.csv"))
    RBVars.Hₙ = (RBVars.Φₛᵘ)' * H
  else
    error("Unrecognized variable to load")
  end

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::PoissonSGRB{T}) where T

  if RBInfo.save_offline_structures
    save_CSV(reshape(RBVars.Aₙ, :, RBVars.Qᵃ)::Matrix{T},
      joinpath(RBInfo.Paths.ROM_structures_path, "Aₙ.csv"))
    if !RBInfo.build_parametric_RHS
      save_CSV(RBVars.Fₙ, joinpath(RBInfo.Paths.ROM_structures_path, "Fₙ.csv"))
      save_CSV(RBVars.Hₙ, joinpath(RBInfo.Paths.ROM_structures_path, "Hₙ.csv"))
    end
  end

end

function get_Q(
  RBInfo::Info,
  RBVars::PoissonSGRB)

  if RBVars.Qᵃ == 0
    RBVars.Qᵃ = size(RBVars.Aₙ)[end]
  end
  if !RBInfo.build_parametric_RHS
    if RBVars.Qᶠ == 0
      RBVars.Qᶠ = size(RBVars.Fₙ)[end]
    end
    if RBVars.Qʰ == 0
      RBVars.Qʰ = size(RBVars.Hₙ)[end]
    end
  end

  return

end

function get_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSGRB,
  Param::ParametricInfoSteady)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    blocks = [1]
    operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

    θᵃ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBVars, θᵃ)
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        get_RB_RHS_blocks(RBVars, θᶠ, θʰ)
      else
        build_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,blocks,blocks,operators)

end

function build_param_RHS(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSGRB,
  Param::ParametricInfoSteady)

  println("Assembling reduced RHS exactly")

  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  Fₙ = RBVars.Φₛᵘ'*(F*RBVars.Φₜᵘ)
  Hₙ = RBVars.Φₛᵘ'*(H*RBVars.Φₜᵘ)

  push!(RBVars.RHSₙ, reshape(Fₙ'+Hₙ',:,1))::Vector{Matrix{T}}

end

function get_θ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSGRB{T},
  Param::ParametricInfoSteady) where T

  θᵃ = get_θᵃ(FEMSpace, RBInfo, RBVars, Param)
  if !RBInfo.build_parametric_RHS
    θᶠ, θʰ = get_θᶠʰ(FEMSpace, RBInfo, RBVars, Param)
  else
    θᶠ, θʰ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  end

  return θᵃ, θᶠ, θʰ

end
