################################# OFFLINE ######################################

function check_norm_matrix(RBVars::PoissonS)
  isempty(RBVars.Xᵘ₀)
end

function PODs_space(
  RBInfo::Info,
  RBVars::PoissonS)

  println("Performing the spatial POD for field u, using a tolerance of $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars)
  RBVars.Φₛᵘ = POD(RBVars.Sᵘ, RBInfo.ϵₛ, RBVars.Xᵘ₀)
  (RBVars.Nₛᵘ, RBVars.nₛᵘ) = size(RBVars.Φₛᵘ)

end

function get_generalized_coordinates(
  RBInfo::ROMInfoS,
  RBVars::PoissonS,
  snaps=nothing)

  get_norm_matrix(RBInfo, RBVars)

  if isnothing(snaps) || maximum(snaps) > RBInfo.nₛ
    snaps = 1:RBInfo.nₛ
  end

  Φₛᵘ_normed = RBVars.Xᵘ₀*RBVars.Φₛᵘ
  RBVars.û = RBVars.Sᵘ[:,snaps]*Φₛᵘ_normed

  if RBInfo.save_offline_structures
    save_CSV(RBVars.û, joinpath(RBInfo.ROM_structures_path, "û.csv"))
  end

end

function set_operators(
  RBInfo::Info,
  ::PoissonS)

  operators = ["A"]
  if !RBInfo.online_RHS
    append!(operators, ["F","H","L"])
  end
  operators

end

function get_A(
  RBInfo::Info,
  RBVars::PoissonS{T}) where T

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
  RBVars::PoissonS{T}) where T

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
  RBVars::PoissonS{T}) where T

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
  RBVars::PoissonS{T}) where T

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
  RBVars::PoissonS{T},
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

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoS,
  RBVars::PoissonS,
  var::String)

  println("The matrix $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")
  if var == "A"
    if isempty(RBVars.MDEIM_mat_A)
      (RBVars.MDEIM_mat_A, RBVars.MDEIM_idx_A, RBVars.MDEIMᵢ_A,
      RBVars.row_idx_A,RBVars.sparse_el_A) = MDEIM_offline(RBInfo, RBVars, "A")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_A, RBVars.row_idx_A, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonS{T},
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

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::PoissonS{T},
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

function assemble_DEIM_vectors(
  RBInfo::ROMInfoS,
  RBVars::PoissonS,
  var::String)

  println("The vector $var is non-affine:
    running the DEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "F"
    if isempty(RBVars.DEIM_mat_F)
      RBVars.DEIM_mat_F, RBVars.DEIM_idx_F, RBVars.DEIMᵢ_F, RBVars.sparse_el_F =
        DEIM_offline(RBInfo,"F")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_F,"F")
  elseif var == "H"
    if isempty(RBVars.DEIM_mat_H)
      RBVars.DEIM_mat_H, RBVars.DEIM_idx_H, RBVars.DEIMᵢ_H, RBVars.sparse_el_H =
        DEIM_offline(RBInfo,"H")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_H,"H")
  elseif var == "L"
    if isempty(RBVars.DEIM_mat_L)
      RBVars.DEIM_mat_L, RBVars.DEIM_idx_L, RBVars.DEIMᵢ_L, RBVars.sparse_el_L =
        DEIM_offline(RBInfo,"L")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_L,"L")
  else
    error("Unrecognized variable on which to perform DEIM")
  end

end

function assemble_reduced_mat_DEIM(
  RBVars::PoissonS{T},
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

function save_assembled_structures(
  RBInfo::Info,
  RBVars::PoissonS)

  affine_vars = (reshape(RBVars.Aₙ, :, RBVars.Qᵃ)::Matrix{T},
                RBVars.Fₙ, RBVars.Hₙ, RBVars.Lₙ)
  affine_names = ("Aₙ", "Fₙ", "Hₙ", "Lₙ")
  save_structures_in_list(affine_vars, affine_names, RBInfo.ROM_structures_path)

  M_DEIM_vars = (RBVars.MDEIM_mat_A, RBVars.MDEIMᵢ_A, RBVars.MDEIM_idx_A,
    RBVars.row_idx_A, RBVars.sparse_el_A, RBVars.DEIM_mat_F, RBVars.DEIMᵢ_F,
    RBVars.DEIM_idx_F, RBVars.sparse_el_F, RBVars.DEIM_mat_H, RBVars.DEIMᵢ_H,
    RBVars.DEIM_idx_H, RBVars.sparse_el_H, RBVars.DEIM_mat_L, RBVars.DEIMᵢ_L,
    RBVars.DEIM_idx_L, RBVars.sparse_el_L)
  M_DEIM_names = (
    "MDEIM_mat_A","MDEIMᵢ_A","MDEIM_idx_A","row_idx_A","sparse_el_A",
    "DEIM_mat_F","DEIMᵢ_F","DEIM_idx_F","sparse_el_F",
    "DEIM_mat_H","DEIMᵢ_H","DEIM_idx_H","sparse_el_H",
    "DEIM_mat_L","DEIMᵢ_L","DEIM_idx_L","sparse_el_L")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path)

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::Info,
  RBVars::PoissonS{T},
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int}) where T

  if !RBInfo.get_offline_structures
    return ["LHS", "RHS"]
  end

  operators = String[]

  for i = LHS_blocks
    LHSₙi = "LHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(RBInfo.ROM_structures_path, LHSₙi * ".csv"))
      append!(operators, ["LHS"])
      break
    end
  end
  for i = RHS_blocks
    RHSₙi = "RHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(RBInfo.ROM_structures_path, RHSₙi * ".csv"))
      append!(operators, ["RHS"])
      break
    end
  end
  if "LHS" ∉ operators
    for i = LHS_blocks
      LHSₙi = "LHSₙ" * string(i) * ".csv"
      println("Importing block number $i of the reduced affine LHS")
      push!(RBVars.LHSₙ,
        load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, LHSₙi)))
    end
  end
  if "RHS" ∉ operators
    for i = RHS_blocks
      RHSₙi = "RHSₙ" * string(i) * ".csv"
      println("Importing block number $i of the reduced affine RHS")
      push!(RBVars.RHSₙ,
        load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, RHSₙi)))
    end
  end

  operators

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::PoissonS{T},
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String}) where T

  if "A" ∉ RBInfo.probl_nl && "LHS" ∈ operators
    for i = LHS_blocks
      LHSₙi = "LHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.LHSₙ[i],joinpath(RBInfo.ROM_structures_path, LHSₙi))
    end
  end
  if "F" ∉ RBInfo.probl_nl && "H" ∉ RBInfo.probl_nl && "RHS" ∈ operators
    for i = RHS_blocks
      RHSₙi = "RHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.RHSₙ[i],joinpath(RBInfo.ROM_structures_path, RHSₙi))
    end
  end
end

function get_θ_matrix(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::PoissonS,
  Param::SteadyParametricInfo,
  var::String) where T

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param.α, RBVars.MDEIMᵢ_A,
      RBVars.MDEIM_idx_A, RBVars.sparse_el_A, "A")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_θ_vector(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::PoissonS,
  Param::SteadyParametricInfo,
  var::String) where T

  if var == "F"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.f, RBVars.DEIMᵢ_F,
      RBVars.DEIM_idx_F, RBVars.sparse_el_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.h, RBVars.DEIMᵢ_H,
      RBVars.DEIM_idx_H, RBVars.sparse_el_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param.g, RBVars.DEIMᵢ_L,
      RBVars.DEIM_idx_L, RBVars.sparse_el_L, "L")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_Q(
  RBInfo::Info,
  RBVars::PoissonS)

  if RBVars.Qᵃ == 0
    RBVars.Qᵃ = size(RBVars.Aₙ)[end]
  end
  if !RBInfo.online_RHS
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

function assemble_param_RHS(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::PoissonS,
  Param::SteadyParametricInfo)

  println("Assembling reduced RHS exactly")

  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")

  push!(RBVars.RHSₙ, reshape(RBVars.Φₛᵘ' * (F + H - L), :, 1)::Matrix{T})

end
