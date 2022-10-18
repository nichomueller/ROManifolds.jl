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

  FEMSpace, μ = get_FEMμ_info(RBInfo)

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
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T}) where {ID,T}

  println("Computing primal supremizers")

  #= if "B" ∈ RBInfo.probl_nl
    println("Matrix Bᵀ is nonaffine: must assemble constraint_matrix ∀ μ")
    constraint_matrix = assemble_constraint_matrix(RBInfo, RBVars)
  else =#
  println("Loading matrix Bᵀ")
  constraint_matrix = load_CSV(sparse([],[],T[]),
    joinpath(get_FEM_structures_path(RBInfo), "B.csv"))'
  #end

  supr_primal = solve_cholesky(RBVars.X₀[1], constraint_matrix * RBVars.Φₛ[2])
  supr_primal[:,1] /= norm(supr_primal[:,1])

  min_norm = 1e16
  for i = 2:size(supr_primal)[2]

    println("Normalizing primal supremizer $i")

    for j in 1:RBVars.nₛ[1]
      #= supr_primal[:, i] -= dot(supr_primal[:, i], RBVars.Φₛ[1][:, j], RBVars.X₀[1]) /
      norm(RBVars.Φₛ[:, j], RBVars.X₀[1]) * RBVars.Φₛ[1][:, j] =#
      supr_primal[:, i] -= (supr_primal[:, i]' * RBVars.Φₛ[1][:, j])*RBVars.Φₛ[1][:, j]
    end
    for j in 1:i-1
      #= supr_primal[:, i] -= dot(supr_primal[:, i], supr_primal[:, j], RBVars.X₀[1]) /
      norm(supr_primal[:, j], RBVars.X₀[1]) * supr_primal[:, j] =#
      supr_primal[:, i] -= ((supr_primal[:, i]' * supr_primal[:, j])/
        (supr_primal[:, j]' * supr_primal[:, j]) * supr_primal[:, j])
    end

    #supr_norm = norm(supr_primal[:, i], RBVars.X₀[1])
    supr_norm = norm(supr_primal[:, i])
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

################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoS,
  RBVars::StokesS{T}) where T

  get_snapshot_matrix(RBInfo, RBVars.Poisson)

  println("Importing the snapshot matrix for field p,
    number of snapshots considered: $(RBInfo.nₛ)")
  Sᵖ = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
    DataFrame))[:, 1:RBInfo.nₛ]
  println("Dimension of pressure snapshot matrix: $(size(Sᵖ))")
  RBVars.Sᵖ = Sᵖ
  RBVars.Nₛᵖ = size(Sᵖ)[1]

end

function get_norm_matrix(
  RBInfo::Info,
  RBVars::StokesS{T}) where T

  if length(RBVars.X₀) == 0
    println("Importing the norm matrix Xu₀")
    Xu₀ = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "Xu₀.csv"))
    println("Importing the norm matrix Xp₀")
    Xp₀ = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "Xp₀.csv"))

    if RBInfo.use_norm_X
      RBVars.X₀ = [Xu₀, Xp₀]
    else
      RBVars.X₀ = [one(T)*sparse(I,RBVars.Nₛᵘ,RBVars.Nₛᵘ),
                   one(T)*sparse(I,RBVars.Nₛᵖ,RBVars.Nₛᵖ)]
    end

  elseif length(RBVars.X₀) == 1
    println("Importing the norm matrix Xp₀")
    Xp₀ = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "Xp₀.csv"))

    if RBInfo.use_norm_X
      RBVars.X₀ = [RBVars.X₀..., Xp₀]
    else
      RBVars.X₀ = [RBVars.X₀..., one(T)*sparse(I,RBVars.Nₛᵖ,RBVars.Nₛᵖ)]
    end
  end

end

function assemble_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::StokesS)

  RBVars.offline_time += @elapsed begin
    POD_space(RBInfo, RBVars)
    supr_enrichment_space(RBInfo, RBVars)
  end

  if RBInfo.save_offline
    save_CSV(RBVars.Φₛ, joinpath(RBInfo.ROM_structures_path,"Φₛ.csv"))
    save_CSV(RBVars.Φₛᵖ, joinpath(RBInfo.ROM_structures_path,"Φₛᵖ.csv"))
  end

  return

end

function get_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::StokesS{T}) where T

  get_reduced_basis(RBInfo, RBVars.Poisson)

  println("Importing the spatial reduced basis for field p")
  RBVars.Φₛᵖ = load_CSV(Matrix{T}(undef,0,0),
    joinpath(RBInfo.ROM_structures_path, "Φₛᵖ.csv"))
  (RBVars.Nₛᵖ, RBVars.nₛᵖ) = size(RBVars.Φₛᵖ)

end

function get_offline_structures(
  RBInfo::ROMInfoS,
  RBVars::StokesS)

  operators = String[]

  append!(operators, get_A(RBInfo, RBVars))
  append!(operators, get_B(RBInfo, RBVars))

  if !RBInfo.online_RHS
    append!(operators, get_F(RBInfo, RBVars))
    append!(operators, get_H(RBInfo, RBVars))
    append!(operators, get_L(RBInfo, RBVars))
    append!(operators, get_Lc(RBInfo, RBVars))
  end

  operators

end

function assemble_offline_structures(
  RBInfo::ROMInfoS,
  RBVars::StokesS,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo, RBVars)
  end

  RBVars.offline_time += @elapsed begin
    for var ∈ setdiff(operators, RBInfo.probl_nl)
      assemble_affine_structures(RBInfo, RBVars, var)
    end

    for var ∈ intersect(operators, RBInfo.probl_nl)
      assemble_MDEIM_structures(RBInfo, RBVars, var)
    end
  end

  if RBInfo.save_offline
    save_assembled_structures(RBInfo, RBVars, operators)
  end

end

function offline_phase(
  RBInfo::ROMInfoS,
  RBVars::StokesS)

  if RBInfo.get_snapshots
    get_snapshot_matrix(RBInfo, RBVars)
    get_snapshots_success = true
  else
    get_snapshots_success = false
  end

  if RBInfo.get_offline_structures
    get_reduced_basis(RBInfo, RBVars)
    get_basis_success = true
  else
    get_basis_success = false
  end

  if !get_snapshots_success && !get_basis_success
    error("Impossible to assemble the reduced problem if
      neither the snapshots nor the bases can be loaded")
  end

  if get_snapshots_success && !get_basis_success
    println("Failed to import the reduced basis, building it via POD")
    assemble_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.get_offline_structures
    operators = get_offline_structures(RBInfo, RBVars)
    if !isempty(operators)
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

end

################################## ONLINE ######################################

function get_θ(
  FEMSpace::FOMS,
  RBInfo::ROMInfoS,
  RBVars::StokesS{T},
  Param::ParamInfoS) where T

  θᵃ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "A")
  θᵇ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "B")

  if !RBInfo.online_RHS
    θᶠ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "F")
    θʰ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "H")
    θˡ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "L")
    θˡᶜ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "Lc")
  else
    θᶠ, θʰ, θˡ, θˡᶜ = Vector{T}[], Vector{T}[], Vector{T}[], Vector{T}[]
  end

  return θᵃ, θᵇ, θᶠ, θʰ, θˡ, θˡᶜ

end

function get_RB_LHS_blocks(
  RBVars::StokesS{T},
  θᵃ::Vector{Vector{T}},
  θᵇ::Vector{Vector{T}}) where T

  get_RB_LHS_blocks(RBVars.Poisson, θᵃ)

  block₂ = sum(Broadcasting(.*)(RBVars.Bₙ, θᵇ))
  push!(RBVars.LHSₙ, -block₂')::Vector{Matrix{T}}
  push!(RBVars.LHSₙ, block₂)::Vector{Matrix{T}}

end

function get_RB_RHS_blocks(
  RBVars::StokesS{T},
  θᶠ::Vector{Vector{T}},
  θʰ::Vector{Vector{T}},
  θˡ::Vector{Vector{T}},
  θˡᶜ::Vector{Vector{T}}) where T

  get_RB_RHS_blocks(RBVars.Poisson, θᶠ, θʰ, θˡ)

  block₂ = - sum(Broadcasting(.*)(RBVars.Lcₙ, θˡᶜ))
  push!(RBVars.RHSₙ, block₂)::Vector{Matrix{T}}

end

function get_RB_system(
  FEMSpace::FOMS,
  RBInfo::ROMInfoS,
  RBVars::StokesS,
  Param::ParamInfoS)

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  LHS_blocks = [1, 2, 3]
  RHS_blocks = [1, 2]

  RBVars.online_time = @elapsed begin
    operators = get_system_blocks(RBInfo, RBVars, LHS_blocks, RHS_blocks)

    θᵃ, θᵇ, θᶠ, θʰ, θˡ, θˡᶜ = get_θ(FEMSpace, RBInfo, RBVars, Param)

    if "LHS" ∈ operators
      get_RB_LHS_blocks(RBVars, θᵃ, θᵇ)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_RHS
        get_RB_RHS_blocks(RBVars, θᶠ, θʰ, θˡ, θˡᶜ)
      else
        assemble_param_RHS(FEMSpace, RBInfo, RBVars, Param)
      end
    end
  end

  save_system_blocks(RBInfo,RBVars,LHS_blocks,RHS_blocks,operators)

end

function solve_RB_system(
  FEMSpace::FOMS,
  RBInfo,
  RBVars::StokesS,
  Param::ParamInfoS) where T

  get_RB_system(FEMSpace, RBInfo, RBVars, Param)
  println("Solving RB problem via backslash")
  RBVars.online_time += @elapsed begin
    LHSₙ = vcat(hcat(RBVars.LHSₙ[1], RBVars.LHSₙ[2]),
      hcat(RBVars.LHSₙ[3], zeros(T, RBVars.nₛᵖ, RBVars.nₛᵖ)))
    RHSₙ = vcat(RBVars.RHSₙ[1], RBVars.RHSₙ[2])
    xₙ = LHSₙ \ RHSₙ
  end
  println("Condition number of the system's matrix: $(cond(LHSₙ))")

  RBVars.uₙ = xₙ[1:RBVars.nₛᵘ,:]
  RBVars.pₙ = xₙ[RBVars.nₛᵘ+1:end,:]

end

function reconstruct_FEM_solution(RBVars::StokesS)
  reconstruct_FEM_solution(RBVars.Poisson)
  RBVars.p̃ = RBVars.Φₛᵖ * RBVars.pₙ
end

function online_phase(
  RBInfo::ROMInfoS,
  RBVars::StokesS{T},
  param_nbs) where T

  FEMSpace, μ = get_FEMμ_info(RBInfo)

  mean_H1_err = 0.0
  mean_L2_err = 0.0
  mean_pointwise_err_u = zeros(T, RBVars.Nₛᵘ)
  mean_pointwise_err_p = zeros(T, RBVars.Nₛᵖ)
  mean_online_time = 0.0
  mean_reconstruction_time = 0.0

  get_norm_matrix(RBInfo, RBVars)

  ũ_μ = zeros(T, RBVars.Nₛᵘ, length(param_nbs))
  uₙ_μ = zeros(T, RBVars.nₛᵘ, length(param_nbs))
  p̃_μ = zeros(T, RBVars.Nₛᵖ, length(param_nbs))
  pₙ_μ = zeros(T, RBVars.nₛᵖ, length(param_nbs))

  for nb in param_nbs
    println("Considering parameter number: $nb")

    Param = ParamInfo(RBInfo, μ[nb])

    uₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "uₕ.csv"),
      DataFrame))[:, nb]
    pₕ_test = Matrix{T}(CSV.read(joinpath(get_FEM_snap_path(RBInfo), "pₕ.csv"),
      DataFrame))[:, nb]

    solve_RB_system(FEMSpace, RBInfo, RBVars, Param)
    reconstruction_time = @elapsed begin
      reconstruct_FEM_solution(RBVars)
    end
    mean_online_time = RBVars.online_time / length(param_nbs)
    mean_reconstruction_time = reconstruction_time / length(param_nbs)

    H1_err_nb = compute_errors(RBVars, uₕ_test, RBVars.ũ, RBVars.X₀[1])
    mean_H1_err += H1_err_nb / length(param_nbs)
    mean_pointwise_err_u += abs.(uₕ_test - RBVars.ũ) / length(param_nbs)

    L2_err_nb = compute_errors(RBVars, pₕ_test, RBVars.p̃, RBVars.X₀[2])
    mean_L2_err += L2_err_nb / length(param_nbs)
    mean_pointwise_err_p += abs.(pₕ_test - RBVars.p̃) / length(param_nbs)

    ũ_μ[:, nb - param_nbs[1] + 1] = RBVars.ũ
    uₙ_μ[:, nb - param_nbs[1] + 1] = RBVars.uₙ
    p̃_μ[:, nb - param_nbs[1] + 1] = RBVars.p̃
    pₙ_μ[:, nb - param_nbs[1] + 1] = RBVars.pₙ

    println("Online wall time: $(RBVars.online_time) s (snapshot number $nb)")
    println("Relative reconstruction H1-error: $H1_err_nb (snapshot number $nb)")
    println("Relative reconstruction L2-error: $L2_err_nb (snapshot number $nb)")

  end

  string_param_nbs = "params"
  for Param_nb in param_nbs
    string_param_nbs *= "_" * string(Param_nb)
  end
  res_path = joinpath(RBInfo.results_path, string_param_nbs)

  if RBInfo.save_online

    create_dir(res_path)
    save_CSV(ũ_μ, joinpath(res_path, "ũ.csv"))
    save_CSV(uₙ_μ, joinpath(res_path, "uₙ.csv"))
    save_CSV(mean_pointwise_err_u, joinpath(res_path, "mean_point_err_u.csv"))
    save_CSV([mean_H1_err], joinpath(res_path, "H1_err.csv"))
    save_CSV(p̃_μ, joinpath(res_path, "p̃.csv"))
    save_CSV(pₙ_μ, joinpath(res_path, "pₙ.csv"))
    save_CSV(mean_pointwise_err_p, joinpath(res_path, "mean_point_err_p.csv"))
    save_CSV([mean_L2_err], joinpath(res_path, "L2_err.csv"))

    if RBInfo.get_offline_structures
      RBVars.offline_time = NaN
    end

    times = Dict("off_time"=>RBVars.offline_time,
      "on_time"=>mean_online_time,"rec_time"=>mean_reconstruction_time)

    CSV.write(joinpath(res_path, "times.csv"),times)

  end

  pass_to_pp = Dict("res_path"=>res_path, "FEMSpace"=>FEMSpace,
    "mean_point_err_u"=>Float.(mean_pointwise_err_u),
    "mean_point_err_p"=>Float.(mean_pointwise_err_p))

  if RBInfo.post_process
    post_process(RBInfo, pass_to_pp)
  end

end
