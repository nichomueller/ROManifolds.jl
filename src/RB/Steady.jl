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

  Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el =
    load_structures_in_list(("Matᵢ_$(var)", "idx_$(var)", "el_$(var)"),
    (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el), RBInfo.ROM_structures_path)

end

function get_MDEIM_structures(
  RBInfo::ROMInfoS{ID},
  Var::VVariable) where ID

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

function reconstruct_FEM_solution(RBVars::ROMMethodS{ID,T}) where {ID,T}
  println("Reconstructing FEM solution")
  push!(RBVars.x̃, Broadcasting(*)(RBVars.Φₛ, RBVars.xₙ))
  return
end

function online_phase(
  RBInfo::ROMInfoS{ID},
  RBVars::ROMMethodS{ID,T},
  param_nbs::Vector{Int}) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))
  get_norm_matrix(RBInfo, RBVars)

  println("Considering parameter numbers: $param_nbs")
  assemble_solve_reconstruct(FEMSpace, RBInfo, RBVars, μ[param_nbs])
  mean_online_time = RBVars.online_time / length(param_nbs)
  println("Online wall time: $(RBVars.online_time)s ")

  xₕ = get_S_var(RBInfo.unknowns, param_nbs)
  norms = get_norms(xₕ[1])
  err = errors(xₕ, RBVars.x̃, RBVars.X₀, norms)
  mean_err = sum(first.(first.(err))) / length(param_nbs)
  mean_pointwise_err = sum(last.(last.(err))) / length(param_nbs)

  if RBInfo.save_online
    save_online(RBInfo, mean_pointwise_err, mean_err, mean_online_time)
  end

  if RBInfo.post_process
    pp(FEMSpace, RBInfo, mean_pointwise_err)
  end

  return

end

function save_online(
  RBInfo::ROMInfoS{ID},
  mean_pointwise_err::Matrix{T},
  mean_err::T,
  mean_online_time::Float) where {ID,T}

  times = times_dictionary(RBInfo, RBVars.offline_time, mean_online_time)
  CSV.write(joinpath(RBInfo.results_path, "times.csv"), times)

  path_err = joinpath(RBInfo.results_path, "mean_err.csv")
  save_CSV([mean_err], path_err)

  path_pwise_err = joinpath(RBInfo.results_path, "mean_point_err.csv")
  save_CSV(mean_pointwise_err, path_pwise_err)

  return
end

#= function save_online(
  RBInfo::ROMInfoS{ID},
  mean_pointwise_err::Vector{Matrix{T}},
  mean_err::Vector{T},
  mean_online_time::Float) where {ID,T}

  times = times_dictionary(RBInfo, RBVars.offline_time, mean_online_time)
  CSV.write(joinpath(RBInfo.results_path, "times.csv"), times)

  path_err = joinpath(RBInfo.results_path, "mean_err.csv")
  save_CSV(mean_err, path_err)

  save_on(i::Int) = save_online(RBInfo, mean_pointwise_err[i], RBInfo.unknowns[i])
  Broadcasting(save_on)(eachindex(RBInfo.unknowns))
  return
end =#
