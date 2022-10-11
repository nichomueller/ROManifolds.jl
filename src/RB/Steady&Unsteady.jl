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

  operators = RBInfo.problem_structures
  if RBInfo.online_RHS
    setdiff(operators, problem_vectors(RBInfo))
  end

  operators

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

function assemble_MDEIM_structures(
  RBInfo::Info,
  RBVars::RBProblem,
  operators::Vector{String})

  function assemble_MDEIM_structure(Var::MVVariable)

    var = Var.var

    if var ∈ operators
      println("The variable $var is non-affine:
        running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

      if isempty(Var.MDEIM.Mat)
        MDEIM_offline!(Var.MDEIM, RBInfo, RBVars, var)
      end
      assemble_MDEIM_Matₙ(Var; get_Φₛ(RBVars, var), RBVars.Φₛ[1])
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

  Φₛ_left = kwargs
  MDEIM = Vars.MDEIM

  Q = size(MDEIM.Mat)[2]

  Vecₙ = Φₛ_left' * MDEIM.Mat
  Vars.Matₙ = [Matrix{T}(reshape(Vecₙ[:,q], :, 1)) for q = 1:Q];

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
  RBInfo::Info,
  RBVars::RBProblem{T},
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
  RBVars::RBProblem{T},
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

function assemble_θ_var(
  FEMSpace::FEMProblem,
  RBInfo::Info,
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
  FEMSpace::FEMProblem,
  RBInfo::Info,
  RBVars::RBProblem,
  μ::Vector{T}) where T

  operators = set_operators(RBInfo)

  θ_var(Var) = assemble_θ_var(FEMSpace, RBInfo, Var, μ, operators)
  Broadcasting(assemble_θ_var)(RBVars.Vars)

end

function assemble_matricesₙ(
  RBInfo::Info,
  RBVars::RBProblem{T},
  Params::Vector{ParamInfo}) where T

  operators = problem_matrices(RBInfo)
  assemble_termsₙ(RBVars.Vars, Params, operators)::Vector{Matrix{T}}

end

function assemble_vectorsₙ(
  RBInfo::Info,
  RBVars::RBProblem{T},
  Params::Vector{ParamInfo}) where T

  operators = problem_vectors(RBInfo)
  assemble_termsₙ(RBVars.Vars, Params, operators)::Vector{Matrix{T}}

end

function assemble_RB_system(
  FEMSpace::FEMProblem,
  RBInfo::Info,
  RBVars::RBProblem,
  μ::Vector{T})

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  blocks = get_blocks_position(RBVars)

  RBVars.online_time = @elapsed begin
    operators = get_system_blocks(RBInfo, RBVars, blocks...)

    Params = assemble_θ(FEMSpace, RBInfo, RBVars, μ)

    if "LHS" ∈ operators
      println("Assembling reduced LHS")
      assemble_LHSₙ(RBVars, Params)
    end

    if "RHS" ∈ operators
      if !RBInfo.online_RHS
        println("Assembling reduced RHS")
        assemble_RHSₙ(RBVars, Params)
      else
        println("Assembling reduced RHS exactly")
        assemble_RHSₙ(FEMSpace, RBInfo, RBVars, μ)
      end
    end
  end

  save_system_blocks(RBInfo, RBVars, operators, blocks...);

end
