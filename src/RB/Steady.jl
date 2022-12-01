################################# OFFLINE ######################################

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

  if RBInfo.load_offline
    get_RB(RBInfo, RBVars)

    operators = load_offline(RBInfo, RBVars)
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


end

function assemble_vectorsₙ(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}


end

function assemble_function_matricesₙ(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}


end

function assemble_function_vectorsₙ(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}


end

function assemble_RHS(
  FEMSpace::FOMS{D},
  RBInfo::ROMInfoS{ID},
  μ::Vector{T}) where {ID,D,T}


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

  xₕ = get_S_var(RBInfo.unknowns, param_nbs, get_snap_path(RBInfo))
  norms = get_norms(xₕ[1])
  err = errors(xₕ, RBVars.x̃, RBVars.X₀, norms)
  mean_err = sum(first.(first.(err))) / length(param_nbs)
  mean_pointwise_err = sum(last.(last.(err))) / length(param_nbs)

  if RBInfo.save_online
    save_online(RBInfo, RBVars.offline_time,
      mean_pointwise_err, mean_err, mean_online_time)
  end

  if RBInfo.postprocess
    pp(FEMSpace, RBInfo, mean_pointwise_err)
  end

  return

end
