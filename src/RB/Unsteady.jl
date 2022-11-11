################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T}) where {ID,T}

  function get_S_var(var::String)
    println("Importing the snapshot matrix for field $var,
      number of snapshots considered: $(RBInfo.nₛ)")
    S = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_snap_path(RBInfo), "$(var)ₕ.csv"))[:, 1:RBInfo.nₛ*RBVars.Nₜ]
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

  function projection(ξnew::Vector{T}, Φ::Matrix{T})
    sum([projection(ξnew, Φ[:, i]) for i=1:size(Φ)[2]])
  end

  function orth_projection(ξnew::Vector{T}, ξ::Vector{T})
    ξ * (ξnew' * ξ) / (ξ' * ξ)
  end

  function orth_projection(ξnew::Vector{T}, Φ::Matrix{T})
    sum([orth_projection(ξnew, Φ[:, i]) for i=1:size(Φ)[2]])
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
    πₗ = l == 1 ? zeros(T, RBVars.nₜ[1]) : orth_projection(Φₜ[:,l], Φₜ[:,1:l-1])
    dist = norm(Φₜ[:,l] - πₗ)
    println("Distance basis number $l is: $dist")
    if dist ≤ tol
      Φₜ, count = enrich(Φₜ, count, l)
    end
    Φₜ, count
  end

  for l = 1:RBVars.nₜ[2]
    Φₜ, count = loop(Φₜ, count, l)
  end

  println("Added $count time supremizers to Φₜ")

end

function supr_enrichment_time(RBVars::ROMMethodST{ID,T}) where {ID,T}

  time_supremizers(RBVars)

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

  Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.time_idx, Var.MDEIM.el =
    load_structures_in_list(("Matᵢ_$(var)", "idx_$(var)", "time_idx_$(var)", "el_$(var)"),
    (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.time_idx, Var.MDEIM.el), RBInfo.ROM_structures_path)

end

function get_MDEIM_structures(
  RBInfo::ROMInfoST{ID},
  Var::VVariable) where ID

  var = Var.var

  Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.time_idx, Var.MDEIM.el =
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

function assemble_ϕₜϕₜθ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  θ::Vector{Vector{T}},
  var::String) where {ID,T}

  Φₜ_left, Φₜ_right = get_Φₜ(RBVars, var)
  idx, idx₁, idx₂ = 1:RBVars.Nₜ, 2:RBVars.Nₜ, 1:RBVars.Nₜ-1

  ΦₜΦₜθ_vec = Φₜ_by_Φₜ_by_θ(Φₜ_left, Φₜ_right, idx, idx, θ)
  ΦₜΦₜθ₁_vec = Φₜ_by_Φₜ_by_θ(Φₜ_left, Φₜ_right, idx₁, idx₂, θ)

  if var == "M"
    ΦₜΦₜθ_vec /= RBInfo.δt*RBInfo.θ
    ΦₜΦₜθ₁_vec /= RBInfo.δt*RBInfo.θ
  end

  ΦₜΦₜθ_vec, ΦₜΦₜθ₁_vec

end

function assemble_ϕₜϕₜθ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  θ::Function,
  var::String) where {ID,T}

  Φₜ_left, Φₜ_right = get_Φₜ(RBVars, var)
  idx, idx₁, idx₂ = 1:RBVars.Nₜ, 2:RBVars.Nₜ, 1:RBVars.Nₜ-1

  ΦₜΦₜθ_vec = Φₜ_by_Φₜ_by_θ(Φₜ_left, Φₜ_right, idx, idx, θ)
  ΦₜΦₜθ₁_vec = Φₜ_by_Φₜ_by_θ(Φₜ_left, Φₜ_right, idx₁, idx₂, θ)

  if var == "M"
    ΦₜΦₜθ_vec /= RBInfo.δt*RBInfo.θ
    ΦₜΦₜθ₁_vec /= RBInfo.δt*RBInfo.θ
  end

  ΦₜΦₜθ_vec, ΦₜΦₜθ₁_vec

end

function assemble_matricesₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  MVar::MVariable{T},
  MParam::ParamInfo) where {ID,T}

  @assert MVar.var == MParam.var
  println("--> Assembling reduced $(MVar.var)")

  ΦₜΦₜθ, ΦₜΦₜθ₁ = assemble_ϕₜϕₜθ(RBInfo, RBVars, MParam.θ, MParam.var)
  Matₙ = assemble_termsₙ(MVar, ΦₜΦₜθ)::Matrix{T}
  Mat₁ₙ = assemble_termsₙ(MVar, ΦₜΦₜθ₁)::Matrix{T}

  if MParam.var == "B"
    _, ΦₜΦₜθ₁ᵀ = assemble_ϕₜϕₜθ(RBInfo, RBVars, MParam.θ, "Bᵀ")
    Mat₁ₙᵀ = assemble_termsₙ(Matrix{T}.(transpose.(MVar.Matₙ)), ΦₜΦₜθ₁ᵀ)::Matrix{T}
    Mat₁ₙ = [Mat₁ₙ, Mat₁ₙᵀ]
  end

  Matₙ, Mat₁ₙ

end

function assemble_matricesₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  MVars::Vector{MVariable{T}},
  MParams::Vector{<:ParamInfo}) where {ID,T}

  Matsₙ = Broadcasting((MVar, MParam) ->
    assemble_matricesₙ(RBInfo, RBVars, MVar, MParam))(MVars, MParams)

  first.(Matsₙ), last.(Matsₙ)

end

function assemble_matricesₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  lin_Mat_ops = get_linear_matrices(RBInfo)
  MVars = MVariable(RBInfo, RBVars, lin_Mat_ops)
  MParams = ParamInfo(Params, lin_Mat_ops)

  assemble_matricesₙ(RBInfo, RBVars, MVars, MParams)

end

function assemble_ϕₜθ(
  RBVars::ROMMethodST{ID,T},
  θ::Vector{Vector{T}},
  var::String) where {ID,T}

  Φₜ_left, _ = get_Φₜ(RBVars, var)
  Φₜ_by_θ(Φₜ_left, θ)

end

function assemble_vectorsₙ(
  RBVars::ROMMethodST{ID,T},
  VVar::VVariable{T},
  VParam::ParamInfo) where {ID,T}

  @assert VVar.var == VParam.var
  println("--> Assembling reduced $(VVar.var)")

  Φₜθ = assemble_ϕₜθ(RBVars, VParam.θ, VParam.var)
  assemble_termsₙ(VVar, Φₜθ)::Matrix{T}

end

function assemble_vectorsₙ(
  RBVars::ROMMethodST{ID,T},
  VVars::Vector{VVariable{T}},
  VParams::Vector{<:ParamInfo}) where {ID,T}

  Broadcasting((VVar, VParam) ->
    assemble_vectorsₙ(RBVars, VVar, VParam))(VVars, VParams)

end

function assemble_vectorsₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  lin_Vec_ops = intersect(get_linear_vectors(RBInfo), set_operators(RBInfo))
  VVars = VVariable(RBInfo, RBVars, lin_Vec_ops)
  VParams = ParamInfo(Params, lin_Vec_ops)

  assemble_vectorsₙ(RBVars, VVars, VParams)

end

function assemble_function_matricesₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  MVar::MVariable{T},
  MParam::ParamInfo) where {ID,T}

  @assert MVar.var == MParam.var
  println("--> Assembling reduced $(MVar.var)")

  ΦₜΦₜθ, ΦₜΦₜθ₁ = assemble_ϕₜϕₜθ(RBInfo, RBVars, MParam.θ, MParam.var)
  Matₙ = assemble_function_termsₙ(MVar, ΦₜΦₜθ)::Matrix{T}
  Mat₁ₙ = assemble_function_termsₙ(MVar, ΦₜΦₜθ₁)::Matrix{T}

  if MParam.var == "B"
    _, ΦₜΦₜθ₁ᵀ = assemble_ϕₜϕₜθ(RBInfo, RBVars, MParam.θ, "Bᵀ")
    Mat₁ₙᵀ = assemble_function_termsₙ(Matrix{T}.(transpose.(MVar.Matₙ)), ΦₜΦₜθ₁ᵀ)::Matrix{T}
    Mat₁ₙ = [Mat₁ₙ, Mat₁ₙᵀ]
  end

  Matₙ, Mat₁ₙ

end

function assemble_function_matricesₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  MVars::Vector{MVariable{T}},
  MParams::Vector{<:ParamInfo}) where {ID,T}

  Matsₙ = Broadcasting((MVar, MParam) ->
    assemble_function_matricesₙ(RBInfo, RBVars, MVar, MParam))(MVars, MParams)

  first.(Matsₙ), last.(Matsₙ)

end

function assemble_function_matricesₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  nonlin_Mat_ops = get_nonlinear_matrices(RBInfo)
  matrix_Vars = MVariable(RBInfo, RBVars, nonlin_Mat_ops)
  matrix_Params = ParamInfo(Params, nonlin_Mat_ops)

  assemble_function_termsₙ(matrix_Vars, matrix_Params)::Vector{<:Function}

end

function assemble_function_vectorsₙ(
  RBVars::ROMMethodST{ID,T},
  VVar::VVariable{T},
  VParam::ParamInfo) where {ID,T}

  @assert VVar.var == VParam.var
  println("--> Assembling reduced $(VVar.var)")

  Φₜθ = assemble_ϕₜθ(RBVars, VParam.θ, VParam.var)
  assemble_function_termsₙ(VVar, Φₜθ)::Matrix{T}

end

function assemble_function_vectorsₙ(
  RBVars::ROMMethodST{ID,T},
  VVars::Vector{VVariable{T}},
  VParams::Vector{<:ParamInfo}) where {ID,T}

  Broadcasting((VVar, VParam) ->
    assemble_function_vectorsₙ(RBVars, VVar, VParam))(VVars, VParams)

end

function assemble_function_vectorsₙ(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  Params::Vector{<:ParamInfo}) where {ID,T}

  nonlin_Vec_ops = get_nonlinear_vectors(RBInfo)
  vector_Vars = VVariable(RBInfo, RBVars, nonlin_Vec_ops)
  vector_Params = ParamInfo(Params, nonlin_Vec_ops)

  assemble_function_termsₙ(vector_Vars, vector_Params)::Vector{<:Function}

end

function assemble_RHS(
  FEMSpace::FOMST{D},
  RBInfo::ROMInfoST{ID},
  μ::Vector{T}) where {ID,D,T}

  lv = setdiff(get_FEM_vectors(RBInfo), get_nonlinear_vectors(RBInfo))
  ParamVec = ParamInfo(RBInfo, μ, lv)
  assemble_FEM_vector(FEMSpace, RBInfo, ParamVec, get_timesθ(RBInfo))

end

function reconstruct_FEM_solution(RBVars::ROMMethodST{ID,T}) where {ID,T}
  println("Reconstructing FEM solution")

  xₙ = Broadcasting(reshape)(RBVars.xₙ, RBVars.nₜ, RBVars.nₛ)
  ΦₛxₙΦₜ(i) = RBVars.Φₛ[i] * (RBVars.Φₜ[i] * xₙ[i])'

  push!(RBVars.x̃, Broadcasting(ΦₛxₙΦₜ)(eachindex(xₙ)))

  return
end

function online_phase(
  RBInfo::ROMInfoST{ID},
  RBVars::ROMMethodST{ID,T},
  param_nbs::Vector{Int}) where {ID,T}

  function get_S_var(var::String, nb::Int, path::String)
    load_CSV(Matrix{Float}(undef,0,0),
      joinpath(path, "$(var)ₕ.csv"))[:, (nb-1)*RBVars.Nₜ+1:nb*RBVars.Nₜ]
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

  if RBInfo.post_process
    pp(FEMSpace, RBInfo, mean_pointwise_err)
  end

  return

end
