################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T}) where {ID,T}

  function get_S_var(var::String)
    println("Importing the snapshot matrix for field $var,
      number of snapshots considered: $(RBInfo.nₛ)")
    S = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_snap_path(RBInfo), "$(var)ₕ.csv"))[:, 1:RBInfo.nₛ*RBVars.Nₜ]
    println("Dimension of snapshot matrix: $(size(S))")

    S, size(S)[1]
  end

  S_info = Broadcasting(get_S_var)(RBInfo.unknowns)::Vector{Tuple{Matrix{T}, Int}}
  RBVars.S, RBVars.Nₛ = first.(S_info), last.(S_info)
  RBVars.N = RBVars.Nₜ * RBVars.Nₛ

  return

end

function time_supremizers(
  RBVars::ROMMethodST{ID,T},
  tol=1e-2) where {ID,T}

  println("Checking if supremizers in time need to be added")

  Φₜ = RBVars.Φₜ[1]' * RBVars.Φₜ[2]
  count = 0

  function projection(ξnew::Vector{T}, ξ::Vector{T})
    ξ * (ξnew' * ξ)
  end

  function projection(ξnew::Vector{T}, Φ::Vector{Vector{T}})
    sum(Broadcasting(ξold -> orth_projection(ξnew, ξold))(Φ))
  end

  function orth_projection(ξnew::Vector{T}, ξ::Vector{T})
    ξ * (ξnew' * ξ) / (ξ' * ξ)
  end

  function orth_projection(ξnew::Vector{T}, Φ::Vector{Vector{T}})
    sum(Broadcasting(ξold -> orth_projection(ξnew, ξold))(Φ))
  end

  function enrich(Φₜ::Matrix{T}, count::Int, l::Int)
    ξˣ = RBVars.Φₜ[2][:,l]
    ξnew = (ξˣ - projection(ξˣ, RBVars.Φₜ[1]))
    ξnew /= norm(ξnew)
    RBVars.Φₜ[1] = hcat(RBVars.Φₜ[1], ξnew)
    RBVars.nₜ[1] += 1
    Φₜ = hcat(Φₜ, ξnew' * RBVars.Φₜ[2])
    count += 1
    Φₜ, count
  end

  function loop(Φₜ::Matrix{T}, count::Int, l::Int)
    πₗ = l == 1 ? Φₜ[:,1] : orth_projection(Φₜ[:,j], Φₜ[:,1:j-1])
    if norm(Φₜ[:,l] - πₗ) ≤ tol
      Φₜ, count = enrich(Φₜ, count, l)
    end
    Φₜ, count
  end

  for l = 1:RBVars.nₜ[2]
    Φₜ, count = loop(Φₜ, count, l)
  end

  println("Added $count time supremizers to Φₜᵘ")

end

function supr_enrichment_time(RBVars::ROMMethodST{ID,T}) where {ID,T}

  supr = assemble_supremizers_time(RBVars)
  RBVars.Φₜ[1] = hcat(RBVars.Φₜ[1], supr)
  RBVars.nₜ[1] = size(RBVars.Φₜ[1])[2]

end

function assemble_RB_time(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T}) where {ID,T}

  get_norm_matrix(RBInfo, RBVars)

  println("Spatial POD, tolerance: $(RBInfo.ϵₛ)")
  RBVars.offline_time += @elapsed begin

    if RBInfo.time_reduction_technique == "ST-HOSVD"
      S = Broadcasting(*)(RBVars.Φₛ, RBVars.S)
    else
      S = RBVars.S
    end
    S₂ = mode₂_unfolding(S, RBInfo.nₛ)

    RBVars.Φₜ = Broadcasting(S -> POD(S, RBInfo.ϵₜ))(S₂)
  end
  RBVars.nₜ = cols(RBVars.Φₛ)

  if ID == 2 || ID == 3
    supr_enrichment_time(RBVars)
  end

  if RBInfo.save_offline
    save_CSV(RBVars.Φₛ, joinpath(RBInfo.ROM_structures_path,"Φₜ.csv"))
  end

  return

end

function assemble_RB(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T}) where {ID,T}

  assemble_RB_space(RBInfo, RBVars)
  assemble_RB_time(RBInfo, RBVars)

  return

end

function get_RB_time(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T}) where {ID,T}

  println("Importing the temporal reduced basis")

  RBVars.Φₜ = matrix_to_blocks(load_CSV(Matrix{T}[],
    joinpath(RBInfo.ROM_structures_path, "Φₜ.csv")), length(RBInfo.unknowns))
  RBVars.Nₜ, RBVars.nₛ = first(rows(RBVars.Φₛ)), cols(RBVars.Φₛ)

  return

end

function get_RB(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T}) where {ID,T}

  get_RB_space(RBInfo, RBVars)
  get_RB_time(RBInfo, RBVars)

end

function get_MDEIM_structures(
  RBInfo::ROMInfoST{ID},
  Var::MVariable) where ID

  var = Var.var

  Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el =
    load_structures_in_list(("Matᵢ_$(var)", "idx_$(var)", "time_idx_$(var)", "el_$(var)"),
    (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.time_idx, Var.MDEIM.el), RBInfo.ROM_structures_path)

end

function get_MDEIM_structures(
  RBInfo::ROMInfoST{ID},
  Var::VVariable) where ID

  var = Var.var

  Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el =
    load_structures_in_list(("Matᵢ_$(var)", "idx_$(var)", "time_idx_$(var)", "el_$(var)"),
    (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.time_idx, Var.MDEIM.el), RBInfo.ROM_structures_path)

end

function offline_phase(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T}) where {ID,T}

  RBVars.Nₜ = Int(RBInfo.tₗ / RBInfo.δt)

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

function reconstruct_FEM_solution(RBVars::ROMMethodST{ID,T}) where {ID,T}
  println("Reconstructing FEM solution")

  xₙ = vector_to_matrix(RBVars.xₙ, RBVars.nₜ, RBVars.nₛ)
  m = Broadcasting(*)
  push!(RBVars.x̃, m(RBVars.Φₛ, m(RBVars.Φₜᵘ, xₙ)'))

  return
end

function online_phase(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  param_nbs::Vector{Int}) where {ID,T}

  FEMSpace, μ = get_FEMμ_info(RBInfo, Val(get_FEM_D(RBInfo)))
  get_norm_matrix(RBInfo, RBVars)

  println("Considering parameter numbers: $param_nbs")
  assemble_solve_reconstruct(FEMSpace, RBInfo, RBVars, μ[param_nbs])
  mean_online_time = RBVars.online_time / length(param_nbs)
  println("Online wall time: $(RBVars.online_time)s ")

  xₕ = get_S_var(RBInfo.unknowns, param_nbs, get_FEM_snap_path(RBInfo))
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
  RBInfo::ROMInfoST{ID},
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
  RBInfo::ROMInfoST{ID},
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
