function get_Aₙ(
  RBInfo::Info,
  RBVars::ADRSGRB{T}) where T

  get_Aₙ(RBInfo, RBVars.Poisson)

end

function get_Bₙ(
  RBInfo::Info,
  RBVars::ADRSGRB{T}) where T

  if isfile(joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))
    println("Importing reduced affine B matrix")
    Bₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))
    RBVars.Bₙ = reshape(Bₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}
    RBVars.Qᵇ = size(RBVars.Bₙ)[3]
    return [""]
  else
    println("Failed to import Dₙ: must build it")
    return ["B"]
  end

end

function get_Dₙ(
  RBInfo::Info,
  RBVars::ADRSGRB{T}) where T

  if isfile(joinpath(RBInfo.ROM_structures_path, "Dₙ.csv"))
    println("Importing reduced affine D matrix")
    Dₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Dₙ.csv"))
    RBVars.Dₙ = reshape(Dₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}
    RBVars.Qᵈ = size(RBVars.Dₙ)[3]
    return [""]
  else
    println("Failed to import Dₙ: must build it")
    return ["D"]
  end

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::ADRSGRB{T},
  var::String) where T

  if var == "B"
    RBVars.Qᵇ = 1
    println("Assembling affine reduced B")
    B = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "B.csv"))
    RBVars.Bₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
    RBVars.Bₙ[:,:,1] = (RBVars.Φₛᵘ)' * B * RBVars.Φₛᵘ
  elseif var == "D"
      RBVars.Qᵈ = 1
      println("Assembling affine reduced D")
      D = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "A.csv"))
      RBVars.Dₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
      RBVars.Dₙ[:,:,1] = (RBVars.Φₛᵘ)' * D * RBVars.Φₛᵘ
  else
    assemble_affine_matrices(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::ADRSGRB{T},
  MDEIM_mat::Matrix,
  row_idx::Vector) where T

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
  elseif var == "D"
    RBVars.Dₙ = Matₙ
    RBVars.Qᵈ = Q
  else
    error("Unrecognized variable")
  end

end

function assemble_reduced_mat_DEIM(
  RBVars::ADRSGRB{T},
  DEIM_mat::Matrix,
  var::String) where T

  assemble_reduced_mat_DEIM(RBVars.Poisson, DEIM_mat, var)

end

function assemble_offline_structures(
  RBInfo::ROMInfoSteady,
  RBVars::ADRSteady,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  assemble_offline_structures(RBInfo, RBVars.Poisson, operators)

  RBVars.offline_time += @elapsed begin
    for var ∈ intersect(operators, RBInfo.probl_nl)
      assemble_MDEIM_matrices(RBInfo, RBVars, var)
    end

    for var ∈ setdiff(operators, RBInfo.probl_nl)
      assemble_affine_matrices(RBInfo, RBVars, var)
    end
  end

  save_affine_structures(RBInfo, RBVars)
  save_M_DEIM_structures(RBInfo, RBVars)

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::ADRSGRB{T},
  var::String) where T

  assemble_affine_vectors(RBInfo, RBVars.Poisson, var)

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::ADRSGRB{T}) where T

  if RBInfo.save_offline_structures
    save_CSV(reshape(RBVars.Bₙ, :, RBVars.Qᵇ)::Matrix{T},
      joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))
    save_CSV(reshape(RBVars.Dₙ, :, RBVars.Qᵈ)::Matrix{T},
      joinpath(RBInfo.ROM_structures_path, "Dₙ.csv"))
  end

end

function get_affine_structures(
  RBInfo::Info,
  RBVars::ADRSGRB)

  operators = get_affine_structures(RBInfo, RBVars.Poisson)

  if "B" ∉ RBInfo.probl_nl
    append!(operators, get_Bₙ(RBInfo, RBVars))
  end
  if "D" ∉ RBInfo.probl_nl
    append!(operators, get_Dₙ(RBInfo, RBVars))
  end

  operators

end

function get_Q(
  RBInfo::Info,
  RBVars::ADRSGRB)

  if RBVars.Qᵇ == 0
    RBVars.Qᵇ = size(RBVars.Bₙ)[end]
  end
  if RBVars.Qᵈ == 0
    RBVars.Qᵈ = size(RBVars.Dₙ)[end]
  end

  get_Q(RBInfo, RBVars.Poisson)

end

function get_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::ADRSGRB,
  Param::SteadyParametricInfo)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    blocks = [1]
    operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

    θᵃ, θᵇ, θᵈ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBVars, θᵃ, θᵇ, θᵈ)
    end

    if "RHS" ∈ operators
      if !RBInfo.assemble_parametric_RHS
        get_RB_RHS_blocks(RBVars, θᶠ, θʰ)
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
      if "L" ∈ RBInfo.probl_nl
        assemble_RB_lifting(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,blocks,blocks,operators)

end

function assemble_RB_lifting(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::ADRSGRB{T},
  Param::SteadyParametricInfo) where T

  println("Assembling reduced lifting exactly")

  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")
  Lₙ = Matrix{T}[]
  push!(Lₙ, RBVars.Φₛᵘ'*L)
  RBVars.RHSₙ -= Lₙ

end

function assemble_param_RHS(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::ADRSGRB,
  Param::SteadyParametricInfo)

  assemble_param_RHS(FEMSpace, RBInfo, RBVars.Poisson, Param)

end

function get_θ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::ADRSGRB{T},
  Param::SteadyParametricInfo) where T

  θᵃ, θᶠ, θʰ = get_θ(FEMSpace, RBInfo, RBVars.Poisson, Param)
  θᵇ = get_θᵇ(FEMSpace, RBInfo, RBVars, Param)
  θᵈ = get_θᵈ(FEMSpace, RBInfo, RBVars, Param)

  return θᵃ, θᵇ, θᵈ, θᶠ, θʰ

end
