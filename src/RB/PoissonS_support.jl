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

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Aₙ.csv"))

    Aₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Aₙ.csv"))
    RBVars.Aₙ = reshape(Aₙ,RBVars.nₛᵘ,RBVars.nₛᵘ,:)::Array{T,3}

    if "A" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_A.Matᵢ, RBVars.MDEIM_A.idx, RBVars.MDEIM_A.el) =
        load_structures_in_list(("Matᵢ_A", "idx_A", "el_A"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for A: must build them")
    op = ["A"]

  end

  op

end

function get_F(
  RBInfo::Info,
  RBVars::PoissonS{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Fₙ.csv"))

    RBVars.Fₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Fₙ.csv"))

    if "F" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_F.Matᵢ, RBVars.MDEIM_F.idx, RBVars.MDEIM_F.el) =
        load_structures_in_list(("Matᵢ_F", "idx_F", "el_F"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for F: must build them")
    op = ["F"]

  end

  op

end

function get_H(
  RBInfo::Info,
  RBVars::PoissonS{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Hₙ.csv"))

    RBVars.Hₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Hₙ.csv"))

    if "H" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_H.Matᵢ, RBVars.MDEIM_H.idx, RBVars.MDEIM_H.el) =
        load_structures_in_list(("Matᵢ_H", "idx_H", "el_H"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for H: must build them")
    op = ["H"]

  end

  op

end

function get_L(
  RBInfo::Info,
  RBVars::PoissonS{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Lₙ.csv"))

    RBVars.Lₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Lₙ.csv"))

    if "L" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_L.Matᵢ, RBVars.MDEIM_L.idx, RBVars.MDEIM_L.el) =
        load_structures_in_list(("Matᵢ_L", "idx_L", "el_L"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for L: must build them")
    op = ["L"]

  end

  op

end

function assemble_affine_structures(
  RBInfo::Info,
  RBVars::PoissonS{T},
  var::String) where T

  if var == "A"
    println("Assembling affine reduced A")
    A = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "A.csv"))
    RBVars.Aₙ = zeros(T, RBVars.nₛᵘ, RBVars.nₛᵘ, 1)
    RBVars.Aₙ[:,:,1] = (RBVars.Φₛᵘ)' * A * RBVars.Φₛᵘ
    RBVars.Qᵃ = 1
  elseif var == "F"
    println("Assembling affine reduced F")
    F = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "F.csv"))
    RBVars.Fₙ = (RBVars.Φₛᵘ)' * F
    RBVars.Qᶠ = 1
  elseif var == "H"
    println("Assembling affine reduced H")
    H = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "H.csv"))
    RBVars.Hₙ = (RBVars.Φₛᵘ)' * H
    RBVars.Qʰ = 1
  elseif var == "L"
    println("Assembling affine reduced L")
    L = load_CSV(Matrix{T}(undef,0,0), joinpath(get_FEM_structures_path(RBInfo), "L.csv"))
    RBVars.Lₙ = (RBVars.Φₛᵘ)' * L
    RBVars.Qˡ = 1
  else
    error("Unrecognized variable to assemble")
  end

end

function assemble_MDEIM_structures(
  RBInfo::ROMInfoS,
  RBVars::PoissonS,
  var::String)

  println("The matrix $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "A"
    if isempty(RBVars.MDEIM_A.Mat)
      MDEIM_offline!(RBVars.MDEIM_A, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_A, var)
  elseif var == "F"
    if isempty(RBVars.MDEIM_F.Mat)
      MDEIM_offline!(RBVars.MDEIM_F, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_F, var)
  elseif var == "H"
    if isempty(RBVars.MDEIM_H.Mat)
      MDEIM_offline!(RBVars.MDEIM_H, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_H, var)
  elseif var == "L"
    if isempty(RBVars.MDEIM_L.Mat)
      MDEIM_offline!(RBVars.MDEIM_L, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_L, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonS{T},
  MDEIM::MDEIMmS,
  ::String) where T

  Q = size(MDEIM.Mat)[2]
  r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, RBVars.Nₛᵘ)
  MatqΦ = zeros(T,RBVars.Nₛᵘ,RBVars.nₛᵘ,Q)
  @simd for j = 1:RBVars.Nₛᵘ
    Mat_idx = findall(x -> x == j, r_idx)
    MatqΦ[j,:,:] = (MDEIM.Mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
  end

  Matₙ = reshape(RBVars.Φₛᵘ' *
    reshape(MatqΦ,RBVars.Nₛᵘ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}

  RBVars.Aₙ = Matₙ
  RBVars.Qᵃ = Q

end

function assemble_reduced_mat_MDEIM(
  RBVars::PoissonS{T},
  MDEIM::MDEIMvS,
  var::String) where T

  Q = size(MDEIM.Mat)[2]
  Vecₙ = zeros(T,RBVars.nₛᵘ,1,Q)
  @simd for q = 1:Q
    Vecₙ[:,:,q] = RBVars.Φₛᵘ' * Vector{T}(MDEIM.Mat[:, q])
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
  RBVars::PoissonS{T},
  operators::Vector{String}) where T

  affine_vars = (reshape(RBVars.Aₙ, RBVars.nₛᵘ ^ 2, :)::Matrix{T},
    RBVars.Fₙ, RBVars.Hₙ, RBVars.Lₙ)
  affine_names = ("Aₙ", "Fₙ", "Hₙ", "Lₙ")
  affine_entry = get_affine_entries(operators, affine_names)
  save_structures_in_list(affine_vars[affine_entry], affine_names[affine_entry],
    RBInfo.ROM_structures_path)

  M_DEIM_vars = (
    RBVars.MDEIM_A.Matᵢ, RBVars.MDEIM_A.idx, RBVars.MDEIM_A.el,
    RBVars.MDEIM_F.Matᵢ, RBVars.MDEIM_F.idx, RBVars.MDEIM_F.el,
    RBVars.MDEIM_H.Matᵢ, RBVars.MDEIM_H.idx, RBVars.MDEIM_H.el,
    RBVars.MDEIM_L.Matᵢ, RBVars.MDEIM_L.idx, RBVars.MDEIM_L.el)
  M_DEIM_names = (
    "Matᵢ_A","idx_A","el_A",
    "Matᵢ_F","idx_F","el_F",
    "Matᵢ_H","idx_H","el_H",
    "Matᵢ_L","idx_L","el_L")
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
    if !isfile(joinpath(RBInfo.ROM_structures_path, LHSₙi))
      append!(operators, ["LHS"])
      break
    end
  end
  for i = RHS_blocks
    RHSₙi = "RHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(RBInfo.ROM_structures_path, RHSₙi))
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
  if "F" ∉ RBInfo.probl_nl && "H" ∉ RBInfo.probl_nl && "L" ∉ RBInfo.probl_nl && "RHS" ∈ operators
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
  Param::ParamInfoS,
  var::String) where T

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.α, RBVars.MDEIM_A, "A")::Matrix{T}
  elseif var == "F"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.f, RBVars.MDEIM_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.h, RBVars.MDEIM_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.MDEIM_L, "L")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_Q(RBVars::PoissonS)

  RBVars.Qᵃ = size(RBVars.Aₙ)[end]
  RBVars.Qᶠ = size(RBVars.Fₙ)[end]
  RBVars.Qʰ = size(RBVars.Hₙ)[end]
  RBVars.Qˡ = size(RBVars.Lₙ)[end]

end

function assemble_param_RHS(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::PoissonS{T},
  Param::ParamInfoS) where T

  println("Assembling reduced RHS exactly")

  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")

  push!(RBVars.RHSₙ, reshape(RBVars.Φₛᵘ' * (F + H - L), :, 1)::Matrix{T})

end
