################################# OFFLINE ######################################

function set_operators(
  RBInfo::Info,
  RBVars::NavierStokesS)

  append!(["C", "D"], set_operators(RBInfo, RBVars.Stokes))

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

    (RBVars.MDEIM_C.Matᵢ, RBVars.MDEIM_C.idx, RBVars.MDEIM_C.el) =
      load_structures_in_list(("Matᵢ_C", "idx_C", "el_C"),
      (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
      RBInfo.ROM_structures_path)

  else

    println("Failed to import offline structures for C: must build them")
    op = ["C"]

  end

  op

end

function get_D(
  RBInfo::Info,
  RBVars::NavierStokesS{T}) where T

  op = String[]

  if isfile(joinpath(RBInfo.ROM_structures_path, "Dₙ.csv"))

    Dₙ = load_CSV(Matrix{T}(undef,0,0), joinpath(RBInfo.ROM_structures_path, "Dₙ.csv"))
    RBVars.Dₙ = reshape(Dₙ, RBVars.nₛᵘ, RBVars.nₛᵘ, :)::Array{T,3}

    (RBVars.MDEIM_D.Matᵢ, RBVars.MDEIM_D.idx, RBVars.MDEIM_D.el) =
      load_structures_in_list(("Matᵢ_D", "idx_D", "el_D"),
      (Matrix{T}(undef,0,0), Vector{Int}(undef,0), Vector{Int}(undef,0)),
      RBInfo.ROM_structures_path)

  else

    println("Failed to import offline structures for D: must build them")
    op = ["D"]

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

function assemble_affine_structures(
  RBInfo::Info,
  RBVars::NavierStokesS{T},
  var::String) where T

  assemble_affine_structures(RBInfo, RBVars.Stokes, var)

end

function assemble_MDEIM_structures(
  RBInfo::ROMInfoS,
  RBVars::NavierStokesS,
  var::String)

  if var == "C"
    if isempty(RBVars.MDEIM_C.Mat)
      MDEIM_offline!(RBVars.MDEIM_C, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_C, var)
  elseif var == "D"
    if isempty(RBVars.MDEIM_D.Mat)
      MDEIM_offline!(RBVars.MDEIM_D, RBInfo, RBVars, var)
    end
    assemble_reduced_mat_MDEIM(RBVars, RBVars.MDEIM_D, var)
  else
    assemble_MDEIM_structures(RBInfo, RBVars.Stokes, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::NavierStokesS{T},
  MDEIM::MDEIMmS,
  var::String) where T

  if var ∈ ("C", "D")
    Q = size(MDEIM.Mat)[2]
    r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, RBVars.Nₛᵘ)
    MatqΦ = zeros(T,RBVars.Nₛᵘ,RBVars.nₛᵘ,Q)::Array{T,3}
    @simd for j = 1:RBVars.Nₛᵘ
      Mat_idx = findall(x -> x == j, r_idx)
      MatqΦ[j,:,:] = (MDEIM.Mat[Mat_idx,:]' * RBVars.Φₛᵘ[c_idx[Mat_idx],:])'
    end

    Matₙ = reshape(RBVars.Φₛᵘ' *
      reshape(MatqΦ,RBVars.Nₛᵘ,:),RBVars.nₛᵘ,:,Q)::Array{T,3}

    if var == "C"
      RBVars.Cₙ = Matₙ
      RBVars.Qᶜ = Q
    else
      RBVars.Dₙ = Matₙ
      RBVars.Qᵈ = Q
    end

  else
    assemble_reduced_mat_MDEIM(RBVars.Stokes, MDEIM, var)
  end

end

function assemble_reduced_mat_MDEIM(
  RBVars::NavierStokesS{T},
  MDEIM::MDEIMvS,
  var::String) where T

  assemble_reduced_mat_MDEIM(RBVars.Stokes, MDEIM, var)

end

function save_assembled_structures(
  RBInfo::Info,
  RBVars::NavierStokesS{T},
  operators::Vector{String}) where T

  Cₙ = reshape(RBVars.Cₙ, RBVars.nₛᵘ ^ 2, :)::Matrix{T}
  Dₙ = reshape(RBVars.Dₙ, RBVars.nₛᵘ ^ 2, :)::Matrix{T}
  affine_vars, affine_names = (Cₙ, Dₙ), ("Cₙ", "Dₙ")
  affine_entry = get_affine_entries(operators, affine_names)
  save_structures_in_list(affine_vars[affine_entry], affine_names[affine_entry],
    RBInfo.ROM_structures_path)

  M_DEIM_vars = (
    RBVars.MDEIM_C.Matᵢ, RBVars.MDEIM_C.idx, RBVars.MDEIM_C.el,
    RBVars.MDEIM_D.Matᵢ, RBVars.MDEIM_D.idx, RBVars.MDEIM_D.el)
  M_DEIM_names = (
    "Matᵢ_C","idx_C","el_C",
    "Matᵢ_D","idx_D","el_D")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path)

  operators_to_pass = setdiff(operators, ("C", "D"))
  save_assembled_structures(RBInfo, RBVars.Stokes, operators_to_pass)

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::Info,
  RBVars::NavierStokesS{T},
  RHS_blocks::Vector{Int}) where T

  if !RBInfo.get_offline_structures
    return ["RHS"]
  end

  operators = String[]

  for i = RHS_blocks
    RHSₙi = "RHSₙ" * string(i) * ".csv"
    if !isfile(joinpath(RBInfo.ROM_structures_path, RHSₙi))
      append!(operators, ["RHS"])
      break
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
  RBVars::NavierStokesS,
  RHS_blocks::Vector{Int},
  operators::Vector{String})

  if ("F" ∉ RBInfo.probl_nl && "H" ∉ RBInfo.probl_nl && "L" ∉ RBInfo.probl_nl
    && "Lc" ∉ RBInfo.probl_nl && "RHS" ∈ operators)
  for i = RHS_blocks
    RHSₙi = "RHSₙ" * string(i) * ".csv"
    save_CSV(RBVars.RHSₙ[i],joinpath(RBInfo.ROM_structures_path, RHSₙi))
  end
end

end

function get_θ_matrix(
  FEMSpace::FEMProblemS,
  RBInfo::ROMInfoS{T},
  RBVars::NavierStokesS,
  Param::ParamInfoS,
  var::String) where T

  if var == "C"
    θ_function(FEMSpace, RBVars, RBVars.MDEIM_C, "C")
  elseif var == "D"
    θ_function(FEMSpace, RBVars, RBVars.MDEIM_D, "D")
  else
    get_θ_matrix(FEMSpace, RBInfo, RBVars.Stokes, Param, var)
  end

end

function get_Q(RBVars::NavierStokesS)

  RBVars.Qᶜ = size(RBVars.Cₙ)[end]
  RBVars.Qᵈ = size(RBVars.Dₙ)[end]

  get_Q(RBVars.Stokes)

end
