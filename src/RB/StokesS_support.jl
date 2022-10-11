################################# OFFLINE ######################################
function POD_space(
  RBInfo::Info,
  RBVars::StokesS)

  POD_space(RBInfo,RBVars.Poisson)

  println("Spatial POD for field p, tolerance: $(RBInfo.ϵₛ)")
  get_norm_matrix(RBInfo, RBVars)
  RBVars.Φₛᵖ = POD(RBVars.Sᵖ, RBInfo.ϵₛ, RBVars.X₀[2])
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function assemble_constraint_matrix(
  RBInfo::Info,
  RBVars::StokesS{T}) where T

  FEMSpace, μ = get_FOM_info(RBInfo.FEMInfo)

  constraint_matrix = zeros(T, RBVars.Nₛᵘ, RBVars.nₛ)

  for k = 1:RBVars.nₛ
    println("Column number $k, constraint matrix")
    Param = ParamInfo(RBInfo, μ[k])
    B_k = Matrix{T}(assemble_FEM_structure(FEMSpace, RBInfo, Param, "B")')
    constraint_matrix[:, k] = B_k * RBVars.Sᵖ[:, k]
  end

  constraint_matrix

end

function primal_supremizers(
  RBInfo::Info,
  RBVars::StokesS{T}) where T

  println("Computing primal supremizers")

  if "B" ∈ RBInfo.probl_nl
    println("Matrix Bᵀ is nonaffine: must assemble constraint_matrix ∀ μ")
    constraint_matrix = assemble_constraint_matrix(RBInfo, RBVars)
  else
    println("Loading matrix Bᵀ")
    constraint_matrix = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))'
  end

  supr_primal = solve_cholesky(RBVars.X₀[1], constraint_matrix * RBVars.Φₛᵖ)

  min_norm = 1e16
  for i = 1:size(supr_primal)[2]

    println("Normalizing primal supremizer $i")

    for j in 1:RBVars.nₛᵘ
      supr_primal[:, i] -= mydot(supr_primal[:, i], RBVars.Φₛ[:, j], RBVars.X₀[1]) /
      mynorm(RBVars.Φₛ[:, j], RBVars.X₀[1]) * RBVars.Φₛ[:, j]
    end
    for j in 1:i
      supr_primal[:, i] -= mydot(supr_primal[:, i], supr_primal[:, j], RBVars.X₀[1]) /
      mynorm(supr_primal[:, j], RBVars.X₀[1]) * supr_primal[:, j]
    end

    supr_norm = mynorm(supr_primal[:, i], RBVars.X₀[1])
    min_norm = min(supr_norm, min_norm)
    println("Norm supremizers: $supr_norm")
    supr_primal[:, i] /= supr_norm

  end

  println("Primal supremizers enrichment ended with norm: $min_norm")

  supr_primal

end

function supr_enrichment_space(
  RBInfo::Info,
  RBVars::StokesS)

  supr_primal = primal_supremizers(RBInfo, RBVars)
  RBVars.Φₛ = hcat(RBVars.Φₛ, supr_primal)
  RBVars.nₛᵘ = size(RBVars.Φₛ)[2]

end

function set_operators(
  RBInfo::Info,
  RBVars::StokesS)

  operators = vcat(["B"], set_operators(RBInfo, RBVars.Poisson))
  if !RBInfo.online_RHS
    append!(operators, ["Lc"])
  end

  operators

end

function get_A(
  RBInfo::Info,
  RBVars::StokesS)

  get_A(RBInfo, RBVars.Poisson)

end

function get_B(
  RBInfo::Info,
  RBVars::StokesS{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))

    RBVars.Bₙ = load_CSV(Matrix{T}[],
      joinpath(RBInfo.ROM_structures_path, "Bₙ.csv"))

    if "B" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_B.Matᵢ, RBVars.MDEIM_B.idx, RBVars.MDEIM_B.el) =
        load_structures_in_list(("Matᵢ_B", "idx_B", "el_B"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for B: must build them")
    op = ["B"]

  end

  op

end

function get_F(
  RBInfo::Info,
  RBVars::StokesS)

  get_F(RBInfo, RBVars.Poisson)

end

function get_H(
  RBInfo::Info,
  RBVars::StokesS)

  get_H(RBInfo, RBVars.Poisson)

end

function get_L(
  RBInfo::Info,
  RBVars::StokesS)

  get_L(RBInfo, RBVars.Poisson)

end

function get_Lc(
  RBInfo::Info,
  RBVars::StokesS{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Lcₙ.csv"))

    RBVars.Lcₙ = load_CSV(Matrix{T}[],
      joinpath(RBInfo.ROM_structures_path, "Lcₙ.csv"))

    if "Lc" ∈ RBInfo.probl_nl

      (RBVars.MDEIM_Lc.Matᵢ, RBVars.MDEIM_Lc.idx, RBVars.MDEIM_Lc.el) =
        load_structures_in_list(("Matᵢ_Lc", "idx_Lc", "el_Lc"),
        (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
        RBInfo.ROM_structures_path)

    end

  else

    println("Failed to import offline structures for Lc: must build them")
    op = ["Lc"]

  end

  op

end

function assemble_affine_structures(
  RBInfo::Info,
  RBVars::StokesS{T},
  var::String) where T

  if var == "B"
    println("Assembling affine reduced B")
    B = load_CSV(sparse([],[],T[]), joinpath(get_FEM_structures_path(RBInfo), "B.csv"))
    push!(RBVars.Bₙ, (RBVars.Φₛᵖ)' * B * RBVars.Φₛ)
  elseif var == "Lc"
    println("Assembling affine reduced Lc")
    Lc = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_structures_path(RBInfo), "Lc.csv"))
    RBVars.Lcₙ = RBVars.Φₛᵖ' * Lc
  else
    assemble_affine_structures(RBInfo, RBVars.Poisson, var)
  end

end

function assemble_MDEIM_structures(
  RBInfo::ROMInfoS,
  RBVars::StokesS,
  var::String)

  println("The variable  $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "A"
    if isempty(RBVars.MDEIM_A.Mat)
      MDEIM_offline!(RBVars.MDEIM_A, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_A, var)
  elseif var == "B"
    if isempty(RBVars.MDEIM_B.Mat)
      MDEIM_offline!(RBVars.MDEIM_B, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_B, var)
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
  elseif var == "Lc"
    if isempty(RBVars.MDEIM_Lc.Mat)
      MDEIM_offline!(RBVars.MDEIM_Lc, RBInfo, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_Lc, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesS{T},
  MDEIM::MMDEIM,
  var::String) where T

  if var == "B"
    Q = size(MDEIM.Mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, RBVars.Nₛᵖ)

    assemble_VecMatΦ(i) = assemble_ith_row_MatΦ(MDEIM.Mat, RBVars.Φₛ, r_idx, c_idx, i)
    VecMatΦ = Broadcasting(assemble_VecMatΦ)(1:RBVars.Nₛᵖ)::Vector{Matrix{T}}
    MatΦ = Matrix{T}(reduce(vcat, VecMatΦ))::Matrix{T}
    Matₙ = reshape(RBVars.Φₛᵖ' * MatΦ, RBVars.nₛᵖ, :, Q)

    RBVars.Bₙ = [Matₙ[:,:,q] for q = 1:Q]
  else
    assemble_reduced_mat_MDEIM(RBVars.Poisson, MDEIM, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::StokesS{T},
  MDEIM::VMDEIM,
  var::String) where T

  if var == "Lc"
    Q = size(MDEIM.Mat)[2]
    Vecₙ = RBVars.Φₛᵖ' * MDEIM.Mat
    Vecₙ_block = [Matrix{T}(reshape(Vecₙ[:,q], :, 1)) for q = 1:Q]

    RBVars.Lcₙ = Vecₙ_block
  else
    assemble_reduced_mat_MDEIM(RBVars.Poisson, MDEIM, var)
  end

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::StokesS{T},
  operators::Vector{String}) where T

  affine_vars, affine_names = (RBVars.Bₙ, RBVars.Lcₙ), ("Bₙ", "Lcₙ")
  affine_entry = get_affine_entries(operators, affine_names)
  save_structures_in_list(affine_vars[affine_entry], affine_names[affine_entry],
    RBInfo.ROM_structures_path)

  MDEIM_vars = (
    RBVars.MDEIM_B.Matᵢ, RBVars.MDEIM_B.idx, RBVars.MDEIM_B.el,
    RBVars.MDEIM_Lc.Matᵢ, RBVars.MDEIM_Lc.idx, RBVars.MDEIM_Lc.el)
  MDEIM_names = (
    "Matᵢ_B","idx_B","el_B",
    "Matᵢ_Lc","idx_Lc","el_Lc")
  save_structures_in_list(MDEIM_vars, MDEIM_names, RBInfo.ROM_structures_path)

  operators_to_pass = setdiff(operators, ("B", "Lc"))
  save_assembled_structures(RBInfo, RBVars.Poisson, operators_to_pass)

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::Info,
  RBVars::StokesS,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int})

  get_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks)

end

function save_system_blocks(
  RBInfo::Info,
  RBVars::StokesS,
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  if "A" ∉ RBInfo.probl_nl && "B" ∉ RBInfo.probl_nl && "LHS" ∈ operators
    for i = LHS_blocks
      LHSₙi = "LHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.LHSₙ[i],joinpath(RBInfo.ROM_structures_path, LHSₙi))
    end
  end
  if ("F" ∉ RBInfo.probl_nl && "H" ∉ RBInfo.probl_nl && "L" ∉ RBInfo.probl_nl
      && "Lc" ∉ RBInfo.probl_nl && "RHS" ∈ operators)
    for i = RHS_blocks
      RHSₙi = "RHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.RHSₙ[i],joinpath(RBInfo.ROM_structures_path, RHSₙi))
    end
  end

end

function get_θ_matrix(
  FEMSpace::FOMS,
  RBInfo,
  RBVars::StokesS,
  Param::ParamInfoS,
  var::String) where T

  θ = Vector{T}[]
  if var == "A"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.α, RBVars.MDEIM_A, "A")
  elseif var == "B"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.b, RBVars.MDEIM_B, "B")
  elseif var == "F"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.f, RBVars.MDEIM_F, "F")
  elseif var == "H"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.h, RBVars.MDEIM_H, "H")
  elseif var == "L"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.MDEIM_L, "L")
  elseif var == "Lc"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.MDEIM_Lc, "Lc")
  else
    error("Unrecognized variable")
  end

end

function assemble_param_RHS(
  FEMSpace::FOMS,
  RBInfo::ROMInfoS,
  RBVars::StokesS{T},
  Param::ParamInfoS) where T

  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")
  push!(RBVars.RHSₙ, reshape(RBVars.Φₛ' * (F + H - L), :, 1)::Matrix{T})

  Lc = assemble_FEM_structure(FEMSpace, RBInfo, Param, "Lc")
  push!(RBVars.RHSₙ, reshape(- RBVars.Φₛᵖ' * Lc, :, 1)::Matrix{T})

end
