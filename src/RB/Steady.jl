################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T}) where {ID,T}

  function get_S_var(var::String)
    println("Importing the snapshot matrix for field $var,
      number of snapshots considered: $(RBInfo.nₛ)")
    S = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_snap_path(RBInfo), "$(var)ₕ.csv"))[:, 1:RBInfo.nₛ]
    println("Dimension of snapshot matrix: $(size(S))")

    S, size(S)[1]
  end

  S_info = Broadcasting(get_S_var)(RBInfo.unknowns)::Vector{Tuple{Matrix{T}, Int}}
  RBVars.S, RBVars.Nₛ = first.(S_info), last.(S_info)

  return

end

function assemble_RB(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T}) where {ID,T}

  assemble_RB_space(RBInfo, RBVars)

  return

end

function get_RB(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T}) where {ID,T}

  get_RB_space(RBInfo, RBVars)

end

function get_MDEIM_structures(
  RBInfo::ROMInfoS{ID},
  Var::MVariable) where ID

  var = Var.var

  Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el =
    load_structures_in_list(("Matᵢ_$(var)", "idx_$(var)", "el_$(var)"),
    (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el), RBInfo.ROM_structures_path)

end

function get_MDEIM_structures(
  RBInfo::ROMInfoS{ID},
  Var::VVariable) where ID

  var = Var.var

  Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el =
    load_structures_in_list(("Matᵢ_$(var)", "idx_$(var)", "el_$(var)"),
    (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el), RBInfo.ROM_structures_path)

end

function offline_phase(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T}) where {ID,T}

  if RBInfo.get_offline_structures
    get_RB(RBInfo, RBVars)

    operators = get_offline_structures(RBInfo, RBVars)
    if !all(isempty.(operators))
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    get_snapshot_matrix(RBInfo, RBVars)

    println("Building reduced basis via POD")
    assemble_RB(RBInfo, RBVars)

    operators = set_operators(RBInfo)
    assemble_offline_structures(RBInfo, RBVars, operators)
  end

end

################################## ONLINE ######################################

function assemble_matricesₙ(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  lin_Mat_ops = get_linear_matrices(RBInfo)
  matrix_Vars = MVariable(RBInfo, RBVars, lin_Mat_ops)
  matrix_Params = ParamInfo(Params, lin_Mat_ops)
  assemble_termsₙ(matrix_Vars, matrix_Params)::Vector{Matrix{T}}

end

function assemble_vectorsₙ(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  lin_Vec_ops = intersect(get_linear_vectors(RBInfo), set_operators(RBInfo))
  vector_Vars = VVariable(RBInfo, RBVars, lin_Vec_ops)
  vector_Params = ParamInfo(Params, lin_Vec_ops)
  assemble_termsₙ(vector_Vars, vector_Params)::Vector{Matrix{T}}

end

function assemble_function_matricesₙ(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  nonlin_Mat_ops = get_nonlinear_matrices(RBInfo)
  matrix_Vars = MVariable(RBInfo, RBVars, nonlin_Mat_ops)
  matrix_Params = ParamInfo(Params, nonlin_Mat_ops)
  assemble_function_termsₙ(matrix_Vars, matrix_Params)::Vector{<:Function}

end

function assemble_function_vectorsₙ(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  nonlin_Vec_ops = get_nonlinear_vectors(RBInfo)
  vector_Vars = VVariable(RBInfo, RBVars, nonlin_Vec_ops)
  vector_Params = ParamInfo(Params, nonlin_Vec_ops)
  assemble_function_termsₙ(vector_Vars, vector_Params)::Vector{<:Function}

end

function assemble_RHS(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{ID},
  μ::Vector{T}) where {ID,D,T}

  lv = setdiff(get_FEM_vectors(RBInfo), get_nonlinear_vectors(RBInfo))
  ParamVec = ParamInfo(RBInfo, μ, lv)
  assemble_FEM_vector(FEMSpace, RBInfo, ParamVec)

end

function reconstruct_FEM_solution(RBVars::ROMMethodS{ID,T}) where {ID,T}
  println("Reconstructing FEM solution")
  push!(RBVars.x̃, Broadcasting(*)(RBVars.Φₛ, RBVars.xₙ))
  return
end

function online_phase(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  param_nbs::Vector{Int}) where {ID,T}

  function get_S_var(var::String, nb::Int, path::String)
    Snb = load_CSV(Matrix{Float}(undef,0,0),
      joinpath(path, "$(var)ₕ.csv"))[:, nb]
    Matrix{Float}(reshape(Snb, :, 1))
  end

  function get_S_var(vars::Vector{String}, nb::Int, path::String)
    Broadcasting(var -> get_S_var(var, nb, path))(vars)
  end

  function get_S_var(vars::Vector{String}, nbs::Vector{Int}, path::String)
    Broadcasting(nb -> get_S_var(vars, nb, path))(nbs)
  end

  μ = get_μ(RBInfo)
  get_norm_matrix(RBInfo, RBVars)

  println("Considering parameter numbers: $param_nbs")
  assemble_solve_reconstruct(RBInfo, RBVars, μ[param_nbs])
  mean_online_time = RBVars.online_time / length(param_nbs)
  println("Online wall time: $(RBVars.online_time)s ")

  xₕ = get_S_var(RBInfo.unknowns, param_nbs, get_FEM_snap_path(RBInfo))
  norms = get_norms(xₕ[1])
  err = errors(xₕ, RBVars.x̃, RBVars.X₀, norms)
  mean_err = sum(first.(first.(err))) / length(param_nbs)
  mean_pointwise_err = sum(last.(last.(err))) / length(param_nbs)

  if RBInfo.save_online
    save_online(RBInfo, RBVars.offline_time,
      mean_pointwise_err, mean_err, mean_online_time)
  end

  if RBInfo.post_process
    pp(FEMSpace, RBInfo, mean_pointwise_err)
  end

  return

end
