################################# OFFLINE ######################################

function get_norm_matrix(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}

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

function assemble_constraint_matrix(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))
  Params = ParamInfo(RBInfo, μ[1:RBVars.nₛ], "B")
  B = assemble_FEM_matrix(FEMSpace, RBInfo, Params)
  BₖΦₖ(k::Int) = B[k] * RBVars.S[2][:, k]

  Broadcasting(BₖΦₖ)(1:RBInfo.nₛ)

end

function assemble_supremizers(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}

  println("Computing primal supremizers")

  if isaffine(RBInfo, "B")
    println("Loading matrix Bᵀ")
    Bᵀ = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "B.csv"))'
    BₖΦₖ(k) = Bᵀ * RBVars.Φₛ[2][:,k]
    constraint_mat = Broadcasting(BₖΦₖ)(1:RBVars.nₛ[2])
  else
    println("Matrix Bᵀ is nonaffine: must assemble the constraint matrix")
    constraint_mat = assemble_constraint_matrix(RBInfo, RBVars)
  end

  supr = Broadcasting(x->solve_cholesky(RBVars.X₀[1], x))(constraint_mat)
  supr_GS = Gram_Schmidt(supr, RBVars.Φₛ[1], RBVars.X₀[1])

  blocks_to_matrix(supr_GS)
end

function supr_enrichment(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}

  supr = assemble_supremizers(RBInfo, RBVars)
  RBVars.Φₛ[1] = hcat(RBVars.Φₛ[1], supr)
  RBVars.nₛ[1] = size(RBVars.Φₛ[1])[2]

end

function assemble_RB_space(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}

  get_norm_matrix(RBInfo, RBVars)

  println("Spatial POD, tolerance: $(RBInfo.ϵₛ)")
  RBVars.offline_time += @elapsed begin
    PODϵ(S, X) = POD(S, RBInfo.ϵₛ, X)
    RBVars.Φₛ = Broadcasting(PODϵ)(RBVars.S, RBVars.X₀)
  end
  RBVars.Nₛ, RBVars.nₛ = rows(RBVars.Φₛ), cols(RBVars.Φₛ)

  if ID == 2 || ID == 3
    supr_enrichment(RBInfo, RBVars)
  end

  if RBInfo.save_offline
    save_CSV(RBVars.Φₛ, joinpath(RBInfo.ROM_structures_path,"Φₛ.csv"))
  end

  return

end

function get_RB_space(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}

  println("Importing the spatial reduced basis")

  RBVars.Φₛ = load_CSV(Matrix{T}[],
    joinpath(RBInfo.ROM_structures_path, "Φₛ.csv"))
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
  RBVars::ROM{ID,T},
  Var::VVariable{T}) where {ID,T}

  var = Var.var

  println("Assembling affine reduced $var")

  function affine_vector(var)
    Φₛ_left, _ = get_Φₛ(RBVars, var)
    Vec = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_structures_path(RBInfo), "$(var).csv"))
    (Φₛ_left' * Vec)
  end

  push!(Var.Matₙ, affine_vector(var)::Matrix{T})

  return

end

function assemble_affine_structure(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  Var::MVariable{T}) where {ID,T}

  var = Var.var

  println("Assembling affine reduced $var")

  function affine_matrix(var)
    Φₛ_left, Φₛ_right = get_Φₛ(RBVars, var)
    Mat = load_CSV(sparse([],[],T[]),
      joinpath(get_FEM_structures_path(RBInfo), "$(var).csv"))
    (Φₛ_left' * Mat * Φₛ_right)
  end

  push!(Var.Matₙ, affine_matrix(var)::Matrix{T})

  return

end

function assemble_affine_structure(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

  Broadcasting(Var -> assemble_affine_structure(RBInfo, RBVars, Var))(Vars)

  return

end

function assemble_MDEIM_structure(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
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
  RBVars::ROM{ID,T},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

  Broadcasting(Var -> assemble_MDEIM_structure(RBInfo, RBVars, Var))(Vars)

  return

end

function assemble_MDEIM_Matₙ(
  Vars::MVariable{T},
  args...) where T

  println("Multiplying MDEIM basis of $(Vars.var) by the RB, this might take some time...")

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

  println("Multiplying MDEIM basis of $(Vars.var) by the RB")

  Φₛ_left, _ = args
  MDEIM = Vars.MDEIM

  Q = size(MDEIM.Mat)[2]

  Vecₙ = Φₛ_left' * MDEIM.Mat
  Vars.Matₙ = [Matrix{T}(reshape(Vecₙ[:,q], :, 1)) for q = 1:Q]

  return

end

function assemble_offline_structures(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  operators::Vector{String}) where {ID,T}

  RBVars.offline_time += @elapsed begin
    am = intersect(operators, get_affine_matrices(RBInfo))
    av = intersect(operators, get_affine_vectors(RBInfo))
    nam = intersect(operators, get_nonaffine_matrices(RBInfo))
    nav = intersect(operators, get_nonaffine_vectors(RBInfo))

    if !isempty(am)
      assemble_affine_structure(RBInfo, RBVars, MVariable(RBInfo, RBVars, am))
    end
    if !isempty(av)
      assemble_affine_structure(RBInfo, RBVars, VVariable(RBInfo, RBVars, av))
    end
    if !isempty(nam)
      assemble_MDEIM_structure(RBInfo, RBVars, MVariable(RBInfo, RBVars, nam))
    end
    if !isempty(nav)
      assemble_MDEIM_structure(RBInfo, RBVars, VVariable(RBInfo, RBVars, nav))
    end
  end

  if RBInfo.save_offline
    Broadcasting(Var -> save_Var_structures(RBInfo, Var, operators))(RBVars.Vars)
  end

  return

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

function save_offline(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  operators::Vector{String}) where {ID,T}


  Broadcasting(Var -> save_Var_structures(RBInfo, Var, operators))(RBVars.Vars)

  return

end

function get_offline_Var(
  RBInfo::ROMInfo{ID},
  Var::MVariable) where ID

  var = Var.var
  println("Importing offline structures for $var")

  Var.Matₙ = load_CSV(Matrix{Float}[],
    joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))

  if var ∉ RBInfo.affine_structures
    get_MDEIM_structures(RBInfo, Var)
  end

end

function get_offline_Var(
  RBInfo::ROMInfo{ID},
  Var::VVariable) where ID

  var = Var.var
  println("Importing offline structures for $var")

  Var.Matₙ = load_CSV(Matrix{Float}[],
    joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))

  if var ∉ RBInfo.affine_structures
    get_MDEIM_structures(RBInfo, Var)
  end

end

function get_offline_Var(
  RBInfo::ROMInfo{ID},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

  Broadcasting(Var -> get_offline_Var(RBInfo, Var))(Vars)

end

function get_offline_structures(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T}) where {ID,T}

  operators = check_saved_operators(RBInfo, RBVars.Vars)::Vector{String}
  operators_to_get = setdiff(set_operators(RBInfo), operators)::Vector{String}
  Vecs_to_get = intersect(get_FEM_vectors(RBInfo), operators_to_get)::Vector{String}
  Mats_to_get = intersect(get_FEM_matrices(RBInfo), operators_to_get)::Vector{String}

  Vars_to_get = vcat(MVariable(RBInfo, RBVars, Mats_to_get),
    VVariable(RBInfo, RBVars, Vecs_to_get))
  get_offline_Var(RBInfo, Vars_to_get)

  operators

end

################################## ONLINE ######################################

function get_system_blocks(
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
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
  RBVars::ROM{ID,T},
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
  μ::Vector{T}) where {ID,D,T}

  var = Var.var
  @assert islinear(RBInfo, var) "Wrong θ assembler"

  Param = ParamInfo(RBInfo, μ, var)
  Param.θ = θ(FEMSpace, RBInfo, Param, Var.MDEIM)
  Param::ParamInfo

end

function assemble_θ(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  μ::Vector{T}) where {ID,D,T}

  lin_Mat_ops = get_linear_matrices(RBInfo)
  MVars = MVariable(RBInfo, RBVars, lin_Mat_ops)
  MParams = Broadcasting(Var -> assemble_θ(FEMSpace, RBInfo, Var, μ))(MVars)
  lin_Vec_ops = get_linear_vectors(RBInfo)
  VVars = VVariable(RBInfo, RBVars, lin_Vec_ops)
  VParams = Broadcasting(Var -> assemble_θ(FEMSpace, RBInfo, Var, μ))(VVars)

  vcat(MParams, VParams)::Vector{<:ParamInfo}

end

function assemble_θ_function(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{ID},
  Var::MVVariable{T},
  μ::Vector{T}) where {ID,D,T}

  var = Var.var
  @assert isnonlinear(RBInfo, var) "Wrong θ assembler"

  Param = ParamInfo(RBInfo, μ, var)
  Param.fun = θ_function(FEMSpace, RBInfo, Param, Var.MDEIM)
  Param::ParamInfo

end

function assemble_θ_function(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  μ::Vector{T}) where {ID,D,T}

  nonlin_Mat_ops = get_nonlinear_matrices(RBInfo)
  MVars = MVariable(RBInfo, RBVars, nonlin_Mat_ops)
  MParams = Broadcasting(Var -> assemble_θ_function(FEMSpace, RBInfo, Var, μ))(MVars)

  MParams::Vector{<:ParamInfo}

end

function assemble_solve_reconstruct(
  FEMSpace::FOM{D},
  RBInfo::ROMInfo{ID},
  RBVars::ROM{ID,T},
  μ::Vector{Vector{T}}) where {ID,D,T}

  Broadcasting(p->assemble_solve_reconstruct(FEMSpace,RBInfo,RBVars,p))(μ)
  return

end

function save_online(
  RBInfo::ROMInfo{ID},
  offline_time::Float,
  mean_pointwise_err::Matrix{T},
  mean_err::T,
  mean_online_time::Float) where {ID,T}

  times = times_dictionary(RBInfo, offline_time, mean_online_time)
  writedlm(joinpath(RBInfo.results_path, "times.csv"), times)

  path_err = joinpath(RBInfo.results_path, "mean_err.csv")
  save_CSV([mean_err], path_err)

  path_pwise_err = joinpath(RBInfo.results_path, "mean_point_err.csv")
  save_CSV(mean_pointwise_err, path_pwise_err)

  return
end
