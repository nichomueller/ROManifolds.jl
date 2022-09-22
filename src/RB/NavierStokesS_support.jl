################################# OFFLINE ######################################

function set_operators(
  RBInfo::Info,
  RBVars::NavierStokesS)

  append!(["C"], set_operators(RBInfo, RBVars.Stokes))

end

function get_A(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_A(RBInfo, RBVars.Stokes)

end

function get_B(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_B(RBInfo, RBVars.Stokes)

end

function get_C(
  RBInfo::Info,
  RBVars::NavierStokesS{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Cₙ.csv"))

    Cₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Cₙ.csv"))
    RBVars.Cₙ = reshape(Cₙ, RBVars.nₛᵘ, RBVars.nₛᵘ, :)::Array{T,3}

    (RBVars.MDEIMᵢ_C, RBVars.MDEIM_idx_C, RBVars.row_idx_C, RBVars.sparse_el_C) =
      load_structures_in_list(("MDEIMᵢ_C", "MDEIM_idx_C", "row_idx_C", "sparse_el_C"),
      (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
      RBInfo.ROM_structures_path)

  else

    println("Failed to import offline structures for C: must build them")
    op = ["C"]

  end

  op

end

function get_F(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_F(RBInfo, RBVars.Stokes)

end

function get_H(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_H(RBInfo, RBVars.Stokes)

end

function get_L(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_L(RBInfo, RBVars.Stokes)

end

function get_Lc(
  RBInfo::Info,
  RBVars::NavierStokesS)

  get_Lc(RBInfo, RBVars.Stokes)

end

function assemble_affine_matrices(
  RBInfo::Info,
  RBVars::NavierStokesS{T},
  var::String) where T

  assemble_affine_matrices(RBInfo, RBVars.Stokes, var)

end

function assemble_MDEIM_matrices(
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS,
  var::String)

  println("The matrix $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if var == "A"
    if isempty(RBVars.MDEIM_mat_A)
      (RBVars.MDEIM_mat_A, RBVars.MDEIM_idx_A, RBVars.MDEIMᵢ_A,
      RBVars.row_idx_A,RBVars.sparse_el_A) = MDEIM_offline(RBInfo, RBVars, "A")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_A, RBVars.row_idx_A, var)
  elseif var == "B"
    if isempty(RBVars.MDEIM_mat_B)
      (RBVars.MDEIM_mat_B, RBVars.MDEIM_idx_B, RBVars.MDEIMᵢ_B,
      RBVars.row_idx_B,RBVars.sparse_el_B) = MDEIM_offline(RBInfo, RBVars, "B")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_B, RBVars.row_idx_B, var)
  elseif var == "C"
    if isempty(RBVars.MDEIM_mat_C)
      (RBVars.MDEIM_mat_C, RBVars.MDEIM_idx_C, RBVars.MDEIMᵢ_C,
      RBVars.row_idx_C,RBVars.sparse_el_C) = MDEIM_offline(RBInfo, RBVars, "C")
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_mat_C, RBVars.row_idx_C, var)
  else
    error("Unrecognized variable on which to perform MDEIM")
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::NavierStokesS{T},
  MDEIM_mat::Matrix,
  row_idx::Vector,
  var::String) where T

  if var == "C"
    Q = size(MDEIM_mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(row_idx, RBVars.Nₛᵘ)
    MatqΦ = zeros(T,RBVars.Nₛᵘ,RBVars.nₛᵘ,Q)::Array{T,3}
    @simd for j = 1:RBVars.Nₛᵘ
      Mat_idx = findall(x -> x == j, r_idx)
      MatqΦ[j,:,:] = (MDEIM_mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
    end

    Matₙ = reshape(RBVars.Φₛᵘ' *
      reshape(MatqΦ,RBVars.Nₛᵘ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}
    RBVars.Cₙ = Matₙ
    RBVars.Qᶜ = Q

  else
    assemble_reduced_mat_MDEIM(RBVars.Stokes, MDEIM_mat, row_idx, var)
  end

end

function assemble_affine_vectors(
  RBInfo::Info,
  RBVars::NavierStokesS,
  var::String)

  assemble_affine_vectors(RBInfo, RBVars.Stokes, var)

end

function assemble_DEIM_vectors(
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS,
  var::String)

  println("The vector $var is non-affine:
    running the DEIM offline phase on $(RBInfo.nₛ_DEIM) snapshots")

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
  elseif var == "Lc"
    if isempty(RBVars.DEIM_mat_Lc)
      RBVars.DEIM_mat_Lc, RBVars.DEIM_idx_Lc, RBVars.DEIMᵢ_Lc, RBVars.sparse_el_Lc =
        DEIM_offline(RBInfo,"Lc")
    end
    assemble_reduced_mat_DEIM(RBVars,RBVars.DEIM_mat_Lc,"Lc")
  else
    error("Unrecognized variable on which to perform DEIM")
  end

end

function assemble_reduced_mat_DEIM(
  RBVars::NavierStokesS,
  DEIM_mat::Matrix,
  var::String)

  assemble_reduced_mat_DEIM(RBVars.Stokes, DEIM_mat, var)

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::StokesS{T},
  operators::Vector{String}) where T

  Cₙ = reshape(RBVars.Cₙ, RBVars.nₛᵘ ^ 2, :)::Matrix{T}
  affine_vars, affine_names = (Cₙ,), ("Cₙ",)
  affine_entry = get_affine_entries(operators, affine_names)
  save_structures_in_list(affine_vars[affine_entry], affine_names[affine_entry],
    RBInfo.ROM_structures_path)

  M_DEIM_vars = (
    RBVars.MDEIMᵢ_C, RBVars.MDEIM_idx_C, RBVars.row_idx_C, RBVars.sparse_el_C)
  M_DEIM_names = (
    "MDEIMᵢ_C","MDEIM_idx_C","row_idx_C","sparse_el_C")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path)

  operators_to_pass = setdiff(operators, ("C",))
  save_assembled_structures(RBInfo, RBVars.Stokes, operators_to_pass)

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

  save_system_blocks(RBInfo, RBVars.Poisson, LHS_blocks, RHS_blocks, operators)

end

function get_θ_matrix(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::NavierStokesS,
  Param::ParamInfoS,
  var::String) where T

  if var == "A"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.α, RBVars.MDEIMᵢ_A,
      RBVars.MDEIM_idx_A, RBVars.sparse_el_A, "A")::Matrix{T}
  elseif var == "B"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, Param.b, RBVars.MDEIMᵢ_B,
      RBVars.MDEIM_idx_B, RBVars.sparse_el_B, "B")::Matrix{T}
  elseif var == "C"
    return θ_matrix(FEMSpace, RBInfo, RBVars, Param, x->1., RBVars.MDEIMᵢ_C,
      RBVars.MDEIM_idx_C, RBVars.sparse_el_C, "C")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_θ_vector(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::NavierStokesS,
  Param::ParamInfoS,
  var::String) where T

  if var == "F"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param, Param.f, RBVars.DEIMᵢ_F,
      RBVars.DEIM_idx_F, RBVars.sparse_el_F, "F")::Matrix{T}
  elseif var == "H"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param, Param.h, RBVars.DEIMᵢ_H,
      RBVars.DEIM_idx_H, RBVars.sparse_el_H, "H")::Matrix{T}
  elseif var == "L"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.DEIMᵢ_L,
      RBVars.DEIM_idx_L, RBVars.sparse_el_L, "L")::Matrix{T}
  elseif var == "Lc"
    return θ_vector(FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.DEIMᵢ_Lc,
      RBVars.DEIM_idx_Lc, RBVars.sparse_el_Lc, "Lc")::Matrix{T}
  else
    error("Unrecognized variable")
  end

end

function get_Q(
  RBInfo::Info,
  RBVars::NavierStokesS)

  RBVars.Qᶜ = size(RBVars.Cₙ)[end]

  get_Q(RBInfo, RBVars.Stokes)

end

function assemble_param_RHS(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS,
  RBVars::StokesS{T},
  Param::ParamInfoS) where T

  F = assemble_FEM_structure(FEMSpace, RBInfo, Param, "F")
  H = assemble_FEM_structure(FEMSpace, RBInfo, Param, "H")
  L = assemble_FEM_structure(FEMSpace, RBInfo, Param, "L")
  push!(RBVars.RHSₙ, reshape(RBVars.Φₛᵘ' * (F + H - L), :, 1)::Matrix{T})

  Lc = assemble_FEM_structure(FEMSpace, RBInfo, Param, "Lc")
  push!(RBVars.RHSₙ, reshape(- RBVars.Φₛᵖ' * Lc, :, 1)::Matrix{T})

end
