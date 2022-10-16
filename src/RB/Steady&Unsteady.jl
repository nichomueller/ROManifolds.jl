################################# OFFLINE ######################################
function get_norm_matrix(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T}) where {ID,T}

  function get_X_var(Nₛ::Int, var::String)
    if RBInfo.use_norm_X
      X₀ = load_CSV(sparse([],[],T[]),
        joinpath(get_FEM_structures_path(RBInfo), "X$(var)₀.csv"))
    else
      X₀ = one(T)*sparse(I, Nₛ, Nₛ)
    end
  end

  if isempty(RBVars.X₀)
    println("Importing the norm matrix")
    RBVars.X₀ = Broadcasting(get_X_var)(RBVars.Nₛ, RBInfo.unknowns)
  end

  return

end

function assemble_reduced_basis_space(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T}) where {ID,T}

  function POD_space(
    S::Matrix,
    ϵₛ::Float,
    X₀::SparseMatrixCSC)

    println("Spatial POD, tolerance: $(ϵₛ)")

    POD(S, ϵₛ, X₀)::Matrix{T}

  end

  get_norm_matrix(RBInfo, RBVars)

  PODϵ(S, X) = POD_space(S, RBInfo.ϵₛ, X)

  RBVars.offline_time += @elapsed begin
    RBVars.Φₛ = Broadcasting(PODϵ)(RBVars.S, RBVars.X₀)
  end
  RBVars.Nₛ, RBVars.nₛ = rows(RBVars.Φₛ), cols(RBVars.Φₛ)

  if RBInfo.save_offline
    save_CSV(RBVars.Φₛ, joinpath(RBInfo.ROM_structures_path,"Φₛ.csv"))
  end

  return

end

function get_reduced_basis_space(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T}) where {ID,T}

  println("Importing the spatial reduced basis")

  RBVars.Φₛ = matrix_to_blocks(load_CSV(Matrix{T}[],
    joinpath(RBInfo.ROM_structures_path, "Φₛ.csv")), length(RBInfo.unknowns))
  RBVars.Nₛ, RBVars.nₛ = rows(RBVars.Φₛ), cols(RBVars.Φₛ)

  return

end

function set_operators(RBInfo::ROMInfo{ID}) where ID

  operators = RBInfo.structures
  if RBInfo.online_RHS
    setdiff(operators, get_FEM_vectors(RBInfo))
  end

  operators::Vector{String}

end

function assemble_affine_structure(
  RBInfo::ROMInfo{ID},
  Var::VVariable{T}) where {ID,T}

  var = Var.var

  println("Assembling affine reduced $var")

  function affine_vector(var)
    Vec = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_structures_path(RBInfo), "$(var).csv"))
    (RBVars.Φₛ' * Vec)
  end

  push!(Var.Matₙ, affine_vector(var)::Matrix{T})

  return

end

function assemble_affine_structure(
  RBInfo::ROMInfo{ID},
  Var::MVariable{T}) where {ID,T}

  var = Var.var

  println("Assembling affine reduced $var")

  function affine_matrix(var)
    Mat = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "$(var).csv"))
    (RBVars.Φₛ' * Mat * RBVars.Φₛ)
  end

  push!(Var.Matₙ, affine_matrix(var)::Matrix{T})

  return

end

function assemble_affine_structure(
  RBInfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

  Broadcasting(Var -> assemble_affine_structure(RBInfo, Var))(Vars)

  return

end

function assemble_MDEIM_structure(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  Var::MVVariable{T}) where {ID,T}

  var = Var.var

  println("The variable $var is non-affine:
    running the MDEIM offline phase on $(RBInfo.nₛ_MDEIM) snapshots")

  if isempty(Var.MDEIM.Mat)
    MDEIM_offline(Var.MDEIM, RBInfo, RBVars, var)
  end
  assemble_MDEIM_Matₙ(Var, get_Φₛ(RBVars, var)...)

  return

end

function assemble_MDEIM_structure(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

  Broadcasting(Var -> assemble_MDEIM_structure(RBInfo, RBVars, Var))(Vars)

  return

end

function assemble_MDEIM_Matₙ(
  Vars::MVariable{T},
  args...) where T

  Φₛ_left, Φₛ_right = args
  MDEIM = Vars.MDEIM

  Q = size(MDEIM.Mat)[2]
  N, n = size(Φₛ_right)[1], size(Φₛ_left)[2]

  r_idx, c_idx = from_vec_to_mat_idx(MDEIM.row_idx, N)

  function assemble_ith_row(i::Int)
    sparse_idx = findall(x -> x == i, r_idx)
    Matrix(reshape((MDEIM.Mat[sparse_idx,:]' * Φₛ_right[c_idx[sparse_idx],:])', 1, :))
  end

  VecMatΦ = Broadcasting(assemble_ith_row)(1:N)::Vector{Matrix{T}}
  MatΦ = Matrix{T}(reduce(vcat, VecMatΦ))::Matrix{T}
  Matₙ = reshape(Φₛ_left' * MatΦ, n, :, Q)

  Vars.Matₙ = [Matₙ[:,:,q] for q = 1:Q]

  return

end

function assemble_MDEIM_Matₙ(
  Vars::VVariable{T},
  args...) where T

  Φₛ_left, _ = args
  MDEIM = Vars.MDEIM

  Q = size(MDEIM.Mat)[2]

  Vecₙ = Φₛ_left' * MDEIM.Mat
  Vars.Matₙ = [Matrix{T}(reshape(Vecₙ[:,q], :, 1)) for q = 1:Q]

  return

end

function assemble_offline_structures(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  operators::Vector{String}) where {ID,T}

  RBVars.offline_time += @elapsed begin
    am = intersect(operators, get_affine_matrices(RBInfo))
    av = intersect(operators, get_affine_vectors(RBInfo))
    nam = intersect(operators, get_nonaffine_matrices(RBInfo))
    nav = intersect(operators, get_nonaffine_vectors(RBInfo))

    assemble_affine_structure(RBInfo, get_matrix_Vars(RBInfo, RBVars, am))
    assemble_affine_structure(RBInfo, get_vector_Vars(RBInfo, RBVars, av))
    assemble_MDEIM_structure(RBInfo, RBVars, get_matrix_Vars(RBInfo, RBVars, nam))
    assemble_MDEIM_structure(RBInfo, RBVars, get_vector_Vars(RBInfo, RBVars, nav))
  end

  if RBInfo.save_offline
    Broadcasting(Var -> save_Var_structures(RBInfo, Var, operators))(RBVars.Vars)
  end

  return

end

function get_offline_structures(
  RBInfo::ROMInfoS{ID},
  RBVars::RBS{T}) where {ID,T}

  operators = check_saved_operators(RBInfo, RBVars.Vars)
  operators_to_get = setdiff(set_operators(RBInfo), operators)
  Vecs_to_get = intersect(get_FEM_vectors(RBInfo), operators_to_get)
  Mats_to_get = intersect(get_FEM_matrices(RBInfo), operators_to_get)

  Vars_to_get = vcat(get_matrix_Vars(RBInfo, RBVars, Mats_to_get),
    get_vector_Vars(RBInfo, RBVars, Vecs_to_get))
  get_offline_Var(RBInfo, Vars_to_get)

  operators

end

function save_Var_structures(
  RBInfo::ROMInfo{ID},
  Var::MVVariable{T},
  operators::Vector{String}) where {ID,T}

  var = Var.var

  if var ∈ operators
    save_CSV(Var.Matₙ, joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))
  end

  MDEIM_vars = (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.time_idx, Var.MDEIM.el)
  MDEIM_names = ("Matᵢ_$(var)", "idx_$(var)", "time_idx_$(var)", "el_$(var)")
  save_structures_in_list(MDEIM_vars, MDEIM_names, RBInfo.ROM_structures_path)

  return

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  LHS_blocks::Vector{Int},
  RHS_blocks::Vector{Int}) where {ID,T}

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
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  operators::Vector{String},
  args...) where {ID,T}

  LHS_blocks, RHS_blocks = args

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

  return

end

function assemble_θ(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{ID},
  Var::MVVariable{T},
  μ::Vector{T}) where {D,ID,T}

  var = Var.var
  Param = ParamInfo(RBInfo, μ, var)
  Param.θ = θ(FEMSpace, RBInfo, Param, Var.MDEIM)

  Param::ParamInfo

end

function assemble_θ(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  μ::Vector{T}) where {D,ID,T}

  Vars = vcat(get_matrix_Vars(RBInfo, RBVars),
    get_vector_Vars(RBInfo, RBVars))
  Broadcasting(Var -> assemble_θ(FEMSpace, RBInfo, Var, μ))(Vars)

end

function assemble_matricesₙ(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  operators = intersect(get_FEM_matrices(RBInfo), set_operators(RBInfo))
  matrix_Vars = get_matrix_Vars(RBInfo, RBVars, operators)
  matrix_Params = ParamInfo(Params, operators)
  assemble_termsₙ(matrix_Vars, matrix_Params)::Vector{Matrix{T}}

end

function assemble_vectorsₙ(
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  operators = intersect(get_FEM_vectors(RBInfo), set_operators(RBInfo))
  vector_Vars = get_vector_Vars(RBInfo, RBVars, operators)
  vector_Params = ParamInfo(Params, operators)
  assemble_termsₙ(vector_Vars, vector_Params)::Vector{Matrix{T}}

end

function assemble_RB_system(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  μ::Vector{T}) where {D,ID,T}

  initialize_RB_system(RBVars)
  initialize_online_time(RBVars)
  blocks = get_blocks_position(RBInfo)

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

  save_system_blocks(RBInfo, RBVars, operators, blocks...)

  return

end

function assemble_solve_reconstruct(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{ID},
  RBVars::RB{T},
  μ::Vector{T}) where {D,ID,T}

  assemble_RB_system(FEMSpace, RBInfo, RBVars, μ)
  RBVars.online_time += @elapsed begin
    solve_RB_system(RBVars)
  end
  reconstruct_FEM_solution(RBVars)

  return

end
