################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoS{ID},
  RBVars::RBS{T}) where {ID,T}

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

function assemble_reduced_basis(
  RBInfo::ROMInfoS{ID},
  RBVars::RBS{T}) where {ID,T}

  assemble_reduced_basis_space(RBInfo, RBVars)

  return

end

function get_reduced_basis(
  RBInfo::ROMInfoS{ID},
  RBVars::RBS{T}) where {ID,T}

  get_reduced_basis_space(RBInfo, RBVars)

end

function get_offline_Var(
  RBInfo::ROMInfoS{ID},
  Var::MVariable) where ID

  var = Var.var
  println("Importing offline structures for $var")

  Matₙ = load_CSV(Matrix{Float}[],
    joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))
  Q = Int(size(Matₙ)[2] / size(Matₙ)[1])
  Var.Matₙ = matrix_to_blocks(Matₙ, Q)

  if var ∉ RBInfo.affine_structures
    Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el =
      load_structures_in_list(("Matᵢ_$(var)", "idx_$(var)", "el_$(var)"),
      (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el), RBInfo.ROM_structures_path)
  end

end

function get_offline_Var(
  RBInfo::ROMInfoS{ID},
  Var::VVariable) where ID

  var = Var.var
  println("Importing offline structures for $var")

  Matₙ = load_CSV(Matrix{Float}[],
    joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))
  Var.Matₙ = matrix_to_blocks(Matₙ)

  if var ∉ RBInfo.affine_structures
    Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el =
      load_structures_in_list(("Matᵢ_$(var)", "idx_$(var)", "el_$(var)"),
      (Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el), RBInfo.ROM_structures_path)
  end

end

function get_offline_Var(
  RBInfo::ROMInfoS{ID},
  Vars::Vector{<:MVVariable{T}}) where {ID,T}

  Broadcasting(Var -> get_offline_Var(RBInfo, Var))(Vars)

end

function save_offline(
  RBInfo::ROMInfoS{ID},
  RBVars::RBS{T},
  operators::Vector{String}) where {ID,T}


  Broadcasting(Var -> save_Var_structures(RBInfo, Var, operators))(RBVars.Vars)

  return

end

function offline_phase(
  RBInfo::ROMInfoS{ID},
  RBVars::RBS{T}) where {ID,T}

  if RBInfo.get_offline_structures
    get_reduced_basis(RBInfo, RBVars)

    operators = get_offline_structures(RBInfo, RBVars)
    if !all(isempty.(operators))
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    get_snapshot_matrix(RBInfo, RBVars)

    println("Building reduced basis via POD")
    assemble_reduced_basis(RBInfo, RBVars)

    operators = set_operators(RBInfo)
    assemble_offline_structures(RBInfo, RBVars, operators)
  end

end

################################## ONLINE ######################################

function reconstruct_FEM_solution(RBVars::RBS{T}) where T
  println("Reconstructing FEM solution")
  RBVars.x̃ = Broadcasting(*)(RBVars.Φₛ, RBVars.xₙ)
  return
end

function online_phase(
  RBInfo::ROMInfoS{ID},
  RBVars::RBS{T},
  param_nbs) where {ID,T}

  function get_S_var(var::String, nb::Int)
    Snb = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_snap_path(RBInfo), "$(var)ₕ.csv"))[:, nb]
    Matrix{T}(reshape(Snb, :, 1))
  end

  function get_norms(solₕ)::Vector{String}
    norms = ["H¹"]
    if length(solₕ) == 2
      push!(norms, ["L²"])
    end
    norms
  end

  function errors(solₕ, sõl, X, norm)
    err_nb = compute_errors(solₕ, sõl, X)
    pointwise_err = abs.(solₕ - sõl)
    println("Online error, norm $norm: $err_nb")
    err_nb, pointwise_err
  end

  function save_online(mean_pointwise_err, mean_err, mean_online_time)
    save_CSV(mean_pointwise_err, joinpath(RBInfo.results_path, "mean_point_err.csv"))
    save_CSV(mean_err, joinpath(RBInfo.results_path, "err.csv"))

    times = times_dictionary(RBInfo, RBVars.offline_time, mean_online_time)
    CSV.write(joinpath(RBInfo.results_path, "times.csv"), times)
    return
  end

  function pp(mean_pointwise_err::Matrix{T})
    pass_to_pp = Dict("res_path"=>RBInfo.results_path, "FEMSpace"=>FEMSpace,
      "mean_point_err"=>Float.(mean_pointwise_err))
    post_process(RBInfo, pass_to_pp)
    return
  end

  function pp(mean_pointwise_err::Vector{Matrix{T}})
    Broadcasting(pp)(mean_pointwise_err)
    return
  end

  FEMSpace, μ = get_FEMμ_info(RBInfo)
  get_norm_matrix(RBInfo, RBVars)

  mean_err = zeros(length(RBVars.Nₛ))
  mean_pointwise_err = Broadcasting(r->zeros(r,1))(RBVars.Nₛ)
  mean_online_time = 0.

  for nb in param_nbs

    println("Considering parameter number: $nb")
    assemble_solve_reconstruct(FEMSpace, RBInfo, RBVars, μ[nb])
    mean_online_time += RBVars.online_time / length(param_nbs)
    get_S(var) = get_S_var(var, nb)
    xₕ = Broadcasting(get_S)(RBInfo.unknowns)::Vector{Vector{Float}}
    norms = get_norms(xₕ)
    errᵢ(i::Int) = errors(xₕ[i], RBVars.x̃[i], RBVars.X₀[i], norms[i])
    err = Broadcasting(errᵢ)(eachindex(xₕ))::Vector{Tuple{Float, Vector{Float}}}
    mean_err += first.(err) / length(param_nbs)
    mean_pointwise_err += last.(err) / length(param_nbs)

    println("Online wall time (snapshot number $nb): $(RBVars.online_time)s ")

  end

  if RBInfo.save_online
    save_online(mean_pointwise_err, mean_err, mean_online_time)
  end

  if RBInfo.post_process
    pp(mean_pointwise_err)
  end

  return

end
