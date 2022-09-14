function get_A(
  RBInfo::Info,
  RBVars::PoissonSGRB{T}) where T

  if "A" ∈ RBInfo.probl_nl

    if isfile(joinpath(RBInfo.ROM_structures_path, "MDEIMᵢ_A.csv"))
      println("Importing MDEIM offline structures, A")
      (RBVars.MDEIMᵢ_A, RBVars.MDEIM_idx_A, RBVars.row_idx_A, RBVars.sparse_el_A) =
        load_structures_in_list(("MDEIMᵢ_A", "MDEIM_idx_A", "row_idx_A", "sparse_el_A"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)
        return [""]
    else
      println("Failed to import MDEIM offline structures for
        A: must build them")
      return ["A"]
    end

  else

    if isfile(joinpath(RBInfo.ROM_structures_path, "Aₙ.csv"))
      println("Importing reduced affine A")
      Aₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Aₙ.csv"))
      RBVars.Aₙ = reshape(Aₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}
      RBVars.Qᵃ = size(RBVars.Aₙ)[3]
      return [""]
    else
      println("Failed to import Aₙ: must build it")
      return ["A"]
    end

  end

end


function get_F(
  RBInfo::Info,
  RBVars::PoissonSteady{T}) where T

  if "F" ∈ RBInfo.probl_nl

    if isfile(joinpath(RBInfo.ROM_structures_path, "DEIMᵢ_F.csv"))
      println("Importing DEIM offline structures, F")
      (RBVars.DEIMᵢ_F, RBVars.DEIM_idx_F, RBVars.sparse_el_F) =
        load_structures_in_list(("DEIMᵢ_F", "DEIM_idx_F", "sparse_el_F"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)
      return [""]
    else
      println("Failed to import DEIM offline structures for F: must build them")
      return ["F"]
    end

  else

    if isfile(joinpath(RBInfo.ROM_structures_path, "Fₙ.csv"))
      println("Importing Fₙ")
      RBVars.Fₙ = load_CSV(Matrix{T}(undef,0,0),
        joinpath(RBInfo.ROM_structures_path, "Fₙ.csv"))
      return [""]
    else
      println("Failed to import Fₙ: must build it")
      return ["F"]
    end

  end

end

function get_H(
  RBInfo::Info,
  RBVars::PoissonSteady{T}) where T

  if "H" ∈ RBInfo.probl_nl

    if isfile(joinpath(RBInfo.ROM_structures_path, "DEIMᵢ_H.csv"))
      println("Importing DEIM offline structures, H")
      (RBVars.DEIMᵢ_H, RBVars.DEIM_idx_H, RBVars.sparse_el_H) =
        load_structures_in_list(("DEIMᵢ_H", "DEIM_idx_H", "sparse_el_H"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)
      return [""]
    else
      println("Failed to import DEIM offline structures for H: must build them")
      return ["H"]
    end

  else

    if isfile(joinpath(RBInfo.ROM_structures_path, "Hₙ.csv"))
      println("Importing Hₙ")
      RBVars.Hₙ = load_CSV(Matrix{T}(undef,0,0),
        joinpath(RBInfo.ROM_structures_path, "Hₙ.csv"))
      return [""]
    else
      println("Failed to import Hₙ: must build it")
      return ["H"]
    end

  end

end

function get_L(
  RBInfo::Info,
  RBVars::PoissonSteady{T}) where T

  if "L" ∈ RBInfo.probl_nl

    if isfile(joinpath(RBInfo.ROM_structures_path, "DEIMᵢ_L.csv"))
      println("Importing DEIM offline structures, L")
      (RBVars.DEIMᵢ_L, RBVars.DEIM_idx_L, RBVars.sparse_el_L) =
        load_structures_in_list(("DEIMᵢ_L", "DEIM_idx_L", "sparse_el_L"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)
      return [""]
    else
      println("Failed to import DEIM offline structures for L: must build them")
      return ["L"]
    end

  else

    if isfile(joinpath(RBInfo.ROM_structures_path, "Lₙ.csv"))
      println("Importing Lₙ")
      RBVars.Lₙ = load_CSV(Matrix{T}(undef,0,0),
        joinpath(RBInfo.ROM_structures_path, "Lₙ.csv"))
      return [""]
    else
      println("Failed to import Lₙ: must build it")
      return ["L"]
    end

  end

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::PoissonSGRB{T},
  var::String) where T

  if var == "A"
    RBVars.Qᵃ = 1
    println("Assembling affine reduced A")
    A = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "A.csv"))
    RBVars.Aₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
    RBVars.Aₙ[:,:,1] = (RBVars.Φₛᵘ)' * A * RBVars.Φₛᵘ
  else
    error("Unrecognized variable to assemble")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonSGRB{T},
  MDEIM_mat::Matrix,
  row_idx::Vector,
  var::String) where T

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
  end

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
  elseif var == "H"
    RBVars.Hₙ = Vecₙ
    RBVars.Qʰ = Q
  else var == "L"
    RBVars.Lₙ = Vecₙ
    RBVars.Qˡ = Q
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::PoissonSGRB{T},
  var::String) where T

  if var == "F"
    RBVars.Qᶠ = 1
    println("Assembling affine reduced forcing term")
    F = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "F.csv"))
    RBVars.Fₙ = (RBVars.Φₛᵘ)' * F
  elseif var == "H"
    RBVars.Qʰ = 1
    println("Assembling affine reduced Neumann term")
    H = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "H.csv"))
    RBVars.Hₙ = (RBVars.Φₛᵘ)' * H
  elseif var == "L"
    RBVars.Qˡ = 1
    println("Assembling affine reduced lifting term")
    L = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "L.csv"))
    RBVars.Lₙ = (RBVars.Φₛᵘ)' * L
  else
    error("Unrecognized variable to assemble")
  end

end

function save_affine_structures(
  RBInfo::Info,
  RBVars::PoissonSGRB{T}) where T

  if RBInfo.save_offline_structures
    save_CSV(reshape(RBVars.Aₙ, :, RBVars.Qᵃ)::Matrix{T},
      joinpath(RBInfo.ROM_structures_path, "Aₙ.csv"))
    if !RBInfo.build_parametric_RHS
      save_CSV(RBVars.Fₙ, joinpath(RBInfo.ROM_structures_path, "Fₙ.csv"))
      save_CSV(RBVars.Hₙ, joinpath(RBInfo.ROM_structures_path, "Hₙ.csv"))
      save_CSV(RBVars.Lₙ, joinpath(RBInfo.ROM_structures_path, "Lₙ.csv"))
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
    if RBVars.Qˡ == 0
      RBVars.Qˡ = size(RBVars.Lₙ)[end]
    end
  end

  return

end

function get_RB_system(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSGRB,
  Param::SteadyParametricInfo)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)

  RBVars.online_time = @elapsed begin
    get_Q(RBInfo, RBVars)
    blocks = [1]
    operators = get_system_blocks(RBInfo, RBVars, blocks, blocks)

    θᵃ, θᶠ, θʰ, θˡ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBVars, θᵃ)
    end

    if "RHS" ∈ operators
      if !RBInfo.build_parametric_RHS
        get_RB_RHS_blocks(RBVars, θᶠ, θʰ, θˡ)
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
  Param::SteadyParametricInfo)

  println("Assembling reduced RHS exactly")

  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")

  push!(RBVars.RHSₙ, reshape(RBVars.Φₛᵘ' * (F + H - L),:,1))::Vector{Matrix{T}}

end

function get_θ(
  FEMSpace::SteadyProblem,
  RBInfo::ROMInfoSteady,
  RBVars::PoissonSGRB{T},
  Param::SteadyParametricInfo) where T

  θᵃ = get_θᵃ(FEMSpace, RBInfo, RBVars, Param)
  if !RBInfo.build_parametric_RHS
    θᶠ, θʰ, θˡ = get_θᶠʰˡ(FEMSpace, RBInfo, RBVars, Param)
  else
    θᶠ, θʰ, θˡ = Matrix{T}(undef,0,0), Matrix{T}(undef,0,0)
  end

  return θᵃ, θᶠ, θʰ, θˡ

end
