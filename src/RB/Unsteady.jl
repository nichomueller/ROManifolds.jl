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

  println("Temporal POD, tolerance: $(RBInfo.ϵₛ)")
  RBVars.offline_time += @elapsed begin

    if RBInfo.t_red_method == "ST-HOSVD"
      ΦₛᵀS(i) = RBVars.Φₛ[i]' * RBVars.S[i]
      S = Broadcasting(ΦₛᵀS)(eachindex(RBVars.S))
    else
      S = RBVars.S
    end
    S₂ = mode₂_unfolding(S, RBInfo.nₛ)

    RBVars.Φₜ = Broadcasting(S -> POD(S, RBInfo.ϵₜ))(S₂)
  end
  RBVars.nₜ = cols(RBVars.Φₜ)

  if ID == 2 || ID == 3
    supr_enrichment_time(RBVars)
  end

  if RBInfo.save_offline
    save_CSV(RBVars.Φₜ, joinpath(RBInfo.ROM_structures_path,"Φₜ.csv"))
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

  RBVars.Φₜ = load_CSV(Matrix{T}[],
    joinpath(RBInfo.ROM_structures_path, "Φₜ.csv"))
  RBVars.nₜ = cols(RBVars.Φₜ)

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

function assemble_ϕₜθ(
  RBVars::ROMMethodST{ID,T},
  Var::MVariable{T},
  Param::ParamInfo) where {ID,T}

  var = Var.var

  Φₜ_left, Φₜ_right = get_Φₜ(RBVars, var)
  nₜ_left, nₜ_right = size(Φₜ_left)[2], size(Φₜ_right)[2]

  Φₜ_by_Φₜ_by_θ(iₜ,jₜ,q,idx1,idx2) =
    sum(Φₜ_left[idx1,iₜ] .* Φₜ_right[idx2,jₜ] .* Param.θ[idx1][q])
  Φₜ_by_Φₜ_by_θ(iₜ,q,idx1,idx2) =
    Broadcasting(jₜ -> Φₜ_by_Φₜ_by_θ(iₜ,jₜ,q,idx1,idx2))(1:nₜ_right)
  Φₜ_by_Φₜ_by_θ(q,idx1,idx2) =
    Broadcasting(z -> Φₜ_by_Φₜ_by_θ(z,q,idx1,idx2))(1:nₜ_left)

  idx1, idx2 = 1:RBVars.Nₜ, 1:RBVars.Nₜ
  ΦₜΦₜθ = Broadcasting(q -> Φₜ_by_Φₜ_by_θ(q,idx1,idx2))(eachindex(Param.θ))
  ΦₜΦₜθ_vec = Broadcasting(blocks_to_matrix)(ΦₜΦₜθ)::Vector{Matrix{T}}

  idx3, idx4 = 2:RBVars.Nₜ, 1:RBVars.Nₜ-1
  ΦₜΦₜθ₁ = Broadcasting(q -> Φₜ_by_Φₜ_by_θ(q,idx3,idx4))(eachindex(Param.θ))
  ΦₜΦₜθ₁_vec = Broadcasting(blocks_to_matrix)(ΦₜΦₜθ₁)::Vector{Matrix{T}}

  hcat(ΦₜΦₜθ_vec, ΦₜΦₜθ₁_vec)::Vector{Matrix{T}}

end

function assemble_ϕₜθ(
  RBVars::ROMMethodST{ID,T},
  Var::VVariable{T},
  Param::ParamInfo) where {ID,T}

  var = Var.var

  Φₜ_left, _ = get_Φₜ(RBVars, var)
  nₜ_left = size(Φₜ_left)[2]

  Φₜ_by_θ(iₜ,q) = sum(Φₜ_left[:,iₜ] .* Param.θ[q])
  Φₜ_by_θ(q) = Broadcasting(iₜ -> Φₜ_by_θ(iₜ,q))(1:nₜ_left)

  Φₜθ = Broadcasting(Φₜ_by_θ)(eachindex(Param.θ))
  blocks_to_matrix(Φₜθ)::Vector{Matrix{T}}

end

function assemble_ϕₜθ(
  RBVars::ROMMethodST{ID,T},
  Vars::Vector{<:MVVariable{T}},
  Params::Vector{<:ParamInfo}) where {ID,T}

  Broadcasting((Var, Param) -> assemble_ϕₜθ(RBVars, Var, Param))(Vars, Params)

end

function assemble_matricesₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  lin_Mat_ops = get_linear_matrices(RBInfo)
  matrix_Vars = MVariable(RBInfo, RBVars, lin_Mat_ops)
  matrix_Params = ParamInfo(Params, lin_Mat_ops)
  ΦₜΦₜθ_all = assemble_ϕₜθ(RBVars, matrix_Vars, matrix_Params)
  ΦₜΦₜθ, ΦₜΦₜθ₁ = first.(ΦₜΦₜθ_all), last.(ΦₜΦₜθ_all)

  Matsₙ = assemble_termsₙ(matrix_Vars, ΦₜΦₜθ)::Vector{Matrix{T}}
  Mats₁ₙ = assemble_termsₙ(matrix_Vars, ΦₜΦₜθ₁)::Vector{Matrix{T}}

  (Matsₙ, Mats₁ₙ)::Tuple{Vector{Matrix{T}}}

end

function assemble_vectorsₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  lin_Vec_ops = intersect(get_linear_vectors(RBInfo), set_operators(RBInfo))
  vector_Vars = VVariable(RBInfo, RBVars, lin_Vec_ops)
  vector_Params = ParamInfo(Params, lin_Vec_ops)
  Φₜθ = assemble_ϕₜθ(RBVars, vector_Vars, vector_Params)

  assemble_termsₙ(vector_Vars, Φₜθ)::Vector{Matrix{T}}

end

function reconstruct_FEM_solution(RBVars::ROMMethodST{ID,T}) where {ID,T}
  println("Reconstructing FEM solution")

  xₙ = vector_to_matrix(RBVars.xₙ, RBVars.nₜ, RBVars.nₛ)
  m = Broadcasting(*)
  push!(RBVars.x̃, m(RBVars.Φₛ, m(RBVars.Φₜ, xₙ)'))

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
