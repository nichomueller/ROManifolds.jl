################################# OFFLINE ######################################
function get_norm_matrix(
  RBInfo::Info,
  RBVars::RBProblem{T}) where T

  function get_X_var(Nₛ::Int, var::String)
    if RBInfo.use_norm_X
      X₀ = load_CSV(sparse([],[],T[]),
        joinpath(get_FEM_structures_path(RBInfo), "X$(var)₀.csv"))
    else
      X₀ = one(T)*sparse(I, Nₛ, Nₛ)
    end
  end

  if isempty(RBVars.X₀[i])
    println("Importing the norm matrix")
    RBVars.Xᵘ = Broadcasting(get_X_var)(RBInfo.vars, RBVars.Nₛ)
  end;

end

function assemble_reduced_basis_space(
  RBInfo::Info,
  RBVars::RBProblem)

  function POD_space(
    S::Matrix,
    ϵₛ::Float,
    X₀::Matrix)

    println("Spatial POD, tolerance: $(ϵₛ)")

    Φₛ = POD(S, ϵₛ, X₀)
    Φₛ, size(Φₛ)...

  end

  RBVars.offline_time += @elapsed begin
    get_norm_matrix(RBInfo, RBVars)
    RBVars.Φₛ, RBVars.Nₛ, RBVars.nₛ =
      Broadcasting(POD_space)(RBVars.S, RBInfo.ϵₛ, RBVars.X)
  end

end

function get_reduced_basis_space(
  RBInfo::Info,
  RBVars::RBProblem) where T

  println("Importing the spatial reduced basis")

  RBVars.Φₛ = load_CSV(Matrix{T}[],
    joinpath(RBInfo.ROM_structures_path, "Φₛ.csv"))
  RBVars.Nₛ, RBVars.nₛ = Broadcasting(get_size)(RBVars.Φₛ);

end

function set_operators(RBInfo::Info)

  operators = RBInfo.FE_matvec
  if RBInfo.online_RHS
    setdiff(operators, RBInfo.FEM_vecs)
  end

  operators

end

function assemble_offline_structures(
  RBInfo::Info,
  RBVars::RBProblem,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo)
  end

  RBVars.offline_time += @elapsed begin
    assemble_affine_structures(RBInfo, RBVars,
      setdiff(operators, RBInfo.probl_nl))
    assemble_MDEIM_structures(RBInfo, RBVars,
      intersect(operators, RBInfo.probl_nl))
  end

end

function assemble_affine_structure(
  RBInfo::Info,
  Var::VVariable{T},
  operators::Vector{String}) where T

  var = Var.var

  if var ∈ operators
    println("Assembling affine reduced $var")

    function affine_vector(var)
      Vec = load_CSV(Matrix{T}(undef,0,0),
        joinpath(get_FEM_structures_path(RBInfo), "$(var).csv"))
      RBVars.Φₛ' * Vec
    end

    Var.Matₙ = affine_vector(var)
  end;

end

function assemble_affine_structure(
  RBInfo::Info,
  Var::MVariable{T},
  operators::Vector{String}) where T

  var = Var.var

  if var ∈ operators
    println("Assembling affine reduced $var")

    function affine_matrix(var)
      Mat = load_CSV(sparse([],[],T[]),
        joinpath(get_FEM_structures_path(RBInfo), "$(var).csv"))
      RBVars.Φₛ' * Mat * RBVars.Φₛ
    end

    Var.Matₙ = affine_matrix(var)
  end;

end

function assemble_affine_structures(
  RBInfo::Info,
  RBVars::RBProblem,
  operators::Vector{String})

  affine_structure(Var) = assemble_affine_structure(RBInfo, Var, operators)
  Broadcasting(affine_structure)(RBVars.Vars);

end

function assemble_MDEIM_structure(
  RBInfo::Info,
  RBVars::RBProblem,
  operators::Vector{String})

  for Var in RBVars.Vars

    var = Var.var

    if var ∈ operators
      println("The variable $var is non-affine:
        running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

      if isempty(Var.MDEIM.Mat)
        MDEIM_offline!(Var.MDEIM, RBInfo, RBVars, var)
      end
      assemble_MDEIM_Matₙ(Var; get_Φₛ(RBVars, var), RBVars.Φₛ[1])
    end

  end;

end

function assemble_MDEIM_structures(
  RBInfo::Info,
  RBVars::RBProblem,
  operators::Vector{String})

  MDEIM_structure(Var) = assemble_MDEIM_structure(RBInfo, Var, operators)
  Broadcasting(MDEIM_structure)(RBVars.Vars);

end

function assemble_MDEIM_Matₙ(
  Vars::MVariable{T};
  kwargs...) where T

  Φₛ_left, Φₛ_right = kwargs
  MDEIM = Vars.MDEIM

  Q = size(MDEIM.Mat)[2]
  N, n = size(Φₛ_right)[1], size(Φₛ_left)[2]

  r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, N)

  assemble_VecMatΦ(i) = assemble_ith_row_MatΦ(MDEIM.Mat, Φₛ_right, r_idx, c_idx, i)
  VecMatΦ = Broadcasting(assemble_VecMatΦ)(1:N)::Vector{Matrix{T}}
  MatΦ = Matrix{T}(reduce(vcat, VecMatΦ))::Matrix{T}
  Matₙ = reshape(Φₛ_left' * MatΦ, n, :, Q)

  Vars.Matₙ = [Matₙ[:,:,q] for q = 1:Q];

end

function assemble_MDEIM_Matₙ(
  Vars::VVariable{T};
  kwargs...) where T

  Φₛ_left = kwargs
  MDEIM = Vars.MDEIM

  Q = size(MDEIM.Mat)[2]

  Vecₙ = Φₛ_left' * MDEIM.Mat
  Vars.Matₙ = [Matrix{T}(reshape(Vecₙ[:,q], :, 1)) for q = 1:Q];

end

function save_Var_structures(
  Var::MVVariable{T},
  operators::Vector{String}) where T

  var = Var.var

  if "$(var)" ∈ operators
    save_structures_in_list(Var.MDEIM.Matₙ, "$(var)ₙ", RBInfo.ROM_structures_path)
  end

  M_DEIM_vars = (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.idx_time, Var.MDEIM.el)
  M_DEIM_names = ("Matᵢ_$(var)", "idx_$(var)", "idx_time_$(var)", "el_$(var)")
  save_structures_in_list(M_DEIM_vars, M_DEIM_names, RBInfo.ROM_structures_path);

end

################################## ONLINE ######################################

function get_θ_matrix(
  FEMSpace::FEMProblem,
  RBInfo::Info,
  RBVars::RBProblem,
  Param::ParamInfo,
  var::String) where T

  θ = Vector{T}[]
  if var == "A"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.α, RBVars.MDEIM_A, "A")
  elseif var == "F"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.f, RBVars.MDEIM_F, "F")
  elseif var == "H"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.h, RBVars.MDEIM_H, "H")
  elseif var == "L"
    θ!(θ, FEMSpace, RBInfo, RBVars, Param, Param.g, RBVars.MDEIM_L, "L")
  else
    error("Unrecognized variable")
  end

end

function get_θ(
  FEMSpace::FEMProblem,
  RBInfo::Info,
  RBVars::RBProblem,
  Param::ParamInfo) where T

  θᵃ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "A")

  if !RBInfo.online_RHS
    θᶠ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "F")
    θʰ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "H")
    θˡ = get_θ_matrix(FEMSpace, RBInfo, RBVars, Param, "L")
  else
    θᶠ, θʰ, θˡ = Vector{T}[], Vector{T}[], Vector{T}[]
  end

  return θᵃ, θᶠ, θʰ, θˡ

end
