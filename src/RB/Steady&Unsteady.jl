################################# OFFLINE ######################################
function get_norm_matrix(
  RBInfo::ROMInfo,
  RBVars::RB{T}) where T

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
    RBVars.Xu = Broadcasting(get_X_var)(RBInfo.vars, RBVars.Nₛ)
  end;

end

function assemble_reduced_basis_space(
  RBInfo::ROMInfo,
  RBVars::RB)

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
  RBInfo::ROMInfo,
  RBVars::RB) where T

  println("Importing the spatial reduced basis")

  RBVars.Φₛ = load_CSV(Matrix{T}[],
    joinpath(RBInfo.ROM_structures_path, "Φₛ.csv"))
  RBVars.Nₛ, RBVars.nₛ = Broadcasting(size)(RBVars.Φₛ);

end

function set_operators(RBInfo::ROMInfo)

  operators = RBInfo.structures
  if RBInfo.online_RHS
    setdiff(operators, get_FEM_vectors(RBInfo))
  end

  operators

end

function assemble_affine_structure(
  RBInfo::ROMInfo,
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

    push!(Var.Matₙ, affine_vector(var))
  end;

end

function assemble_affine_structure(
  RBInfo::ROMInfo,
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

    push!(Var.Matₙ, affine_matrix(var))
  end;

end

function assemble_affine_structures(
  RBInfo::ROMInfo,
  RBVars::RB,
  operators::Vector{String})

  affine_structure(Var) = assemble_affine_structure(RBInfo, Var, operators)
  Broadcasting(affine_structure)(RBVars.Vars);

end

function assemble_MDEIM_structures(
  RBInfo::ROMInfo,
  RBVars::RB,
  operators::Vector{String})

  function assemble_MDEIM_structure(Var::MVVariable)

    var = Var.var

    if var ∈ operators
      println("The variable $var is non-affine:
        running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

      if isempty(Var.MDEIM.Mat)
        MDEIM_offline(Var.MDEIM, RBInfo, RBVars, var)
      end
      assemble_MDEIM_Matₙ(Var; get_Φₛ(RBVars, var)...)
    end;

  end

  Broadcasting(assemble_MDEIM_structure)(RBVars.Vars);

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

  Φₛ_left, _ = kwargs
  MDEIM = Vars.MDEIM

  Q = size(MDEIM.Mat)[2]

  Vecₙ = Φₛ_left' * MDEIM.Mat
  Vars.Matₙ = [Matrix{T}(reshape(Vecₙ[:,q], :, 1)) for q = 1:Q];

end

function assemble_offline_structures(
  RBInfo::ROMInfo,
  RBVars::RB,
  operators=String[])

  if isempty(operators)
    operators = set_operators(RBInfo)
  end

  RBVars.offline_time += @elapsed begin
    assemble_affine_structures(RBInfo, RBVars,
      setdiff(operators, RBInfo.affine_structures))
    assemble_MDEIM_structures(RBInfo, RBVars,
      intersect(operators, RBInfo.affine_structures))
  end

end

function save_Var_structures(
  Var::MVVariable{T},
  operators::Vector{String}) where T

  var = Var.var

  if "$(var)" ∈ operators
    save_structures_in_list(Var.MDEIM.Matₙ, "$(var)ₙ", RBInfo.ROM_structures_path)
  end

  MDEIM_vars = (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.idx_time, Var.MDEIM.el)
  MDEIM_names = ("Matᵢ_$(var)", "idx_$(var)", "idx_time_$(var)", "el_$(var)")
  save_structures_in_list(MDEIM_vars, MDEIM_names, RBInfo.ROM_structures_path);

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::ROMInfo,
  RBVars::RB{T},
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
  RBInfo::ROMInfo,
  RBVars::RB{T},
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int},
  operators::Vector{String}) where T

  if get_FEM_matrices(RBInfo) ∈ RBInfo.affine_structures && "LHS" ∈ operators
    for i = LHS_blocks
      LHSₙi = "LHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.LHSₙ[i],joinpath(RBInfo.ROM_structures_path, LHSₙi))
    end
  end
  if get_FEM_vectors(RBInfo) ∈ RBInfo.affine_structures && "RHS" ∈ operators
    for i = RHS_blocks
      RHSₙi = "RHSₙ" * string(i) * ".csv"
      save_CSV(RBVars.RHSₙ[i],joinpath(RBInfo.ROM_structures_path, RHSₙi))
    end
  end
end

function assemble_θ_var(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  Var::MVVariable,
  μ::Vector{T},
  operators::Vector{String}) where T

  var = Var.var

  if var ∈ operators
    Param = ParamInfo(RBInfo, μ, var)
    Param.θ = θ(FEMSpace, RBInfo, Param, MDEIM)
  else
    Param.θ = Vector{T}[]
  end

  Param::ParamInfo

end

function assemble_θ(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  RBVars::RB,
  μ::Vector{T}) where T

  operators = set_operators(RBInfo)

  θ_var(Var) = assemble_θ_var(FEMSpace, RBInfo, Var, μ, operators)
  Broadcasting(assemble_θ_var)(RBVars.Vars)

end

function assemble_matricesₙ(
  RBInfo::ROMInfo,
  RBVars::RB{T},
  Params::Vector{ParamInfo}) where T

  operators = get_FEM_matrices(RBInfo)
  assemble_termsₙ(RBVars.Vars, Params, operators)::Vector{Matrix{T}}

end

function assemble_vectorsₙ(
  RBInfo::ROMInfo,
  RBVars::RB{T},
  Params::Vector{ParamInfo}) where T

  operators = get_FEM_vectors(RBInfo)
  assemble_termsₙ(RBVars.Vars, Params, operators)::Vector{Matrix{T}}

end

function assemble_RB_system(
  FEMSpace::FOM,
  RBInfo::ROMInfo,
  RBVars::RB,
  μ::Vector{T})

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  blocks = get_blocks_position(RBVars)

  RBVars.online_time = @elapsed begin
    operators = get_system_blocks(RBInfo, RBVars, blocks...)

    Params = assemble_θ(FEMSpace, RBInfo, RBVars, μ)

    if "LHS" ∈ operators
      println("Assembling reduced LHS")
      assemble_LHSₙ(RBInfo, RBVars, Params)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_RHS
        println("Assembling reduced RHS")
        assemble_RHSₙ(RBInfo, RBVars, Params)
      else
        println("Assembling reduced RHS exactly")
        assemble_RHSₙ(FEMSpace, RBInfo, RBVars, μ)
      end
    end
  end

  save_system_blocks(RBInfo, RBVars, operators, blocks...);

end
