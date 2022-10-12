################################# OFFLINE ######################################

function get_snapshot_matrix(
  RBInfo::ROMInfoS,
  RBVars::RBS{T}) where T

  function get_S_var(var::String)
    println("Importing the snapshot matrix for field $var,
      number of snapshots considered: $(RBInfo.nₛ)")
    S = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_snap_path(RBInfo), "$(var)ₕ.csv"))[:, 1:RBInfo.nₛ]
    println("Dimension of snapshot matrix: $(size(S))")

    S, size(S)[1]
  end

  RBVars.S, RBVars.Nₛ = Broadcasting(get_S_var)(RBInfo.unknowns);

end

function assemble_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::RBS)

  assemble_reduced_basis_space(RBInfo, RBVars);

end

function get_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::RBS)

  get_reduced_basis_space(RBInfo, RBVars);

end

function get_offline_structures(
  RBInfo::ROMInfoS,
  RBVars::RBS{T}) where T

  function get_Var(Var::MVVariable)

    var = Var.var

    if !(var ∈ get_FEM_vectors(RBInfo) && RBInfo.online_RHS)
      if isfile(joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))
        Var.Matₙ = load_CSV(Matrix{T}[],
          joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))
        if var ∉ RBInfo.affine_structures
          Var.MDEIM.Matᵢ, Var.MDEIM.idx, Var.MDEIM.el =
            load_structures_in_list(("Matᵢ_$(var)", "idx_$(var)", "el_$(var)"),
            (Matᵢ, idx, el), RBInfo.ROM_structures_path)
        end
      else
        op = var
        println("Failed to import offline structures for $var: must build them")
      end
    end

    Var, op

  end

  RBVars.Vars, operators = Broadcast(get_var)(RBVars.Vars)

  operators

end

function save_offline(
  RBInfo::ROMInfoS,
  RBVars::RBS{T},
  operators::Vector{String}) where T

  save_CSV(RBVars.Φₛ, joinpath(RBInfo.ROM_structures_path,"Φₛ.csv"))

  save_Var(Var) = save_Var_structures(Var, operators)
  Broadcasting(save_Var)(RBVars.Vars);

end

function offline_phase(
  RBInfo::ROMInfoS,
  RBVars::RBS)

  if RBInfo.get_snapshots
    get_snapshot_matrix(RBInfo, RBVars)
    get_snapshots_success = true
  else
    get_snapshots_success = false
  end

  if RBInfo.get_offline_structures
    get_reduced_basis(RBInfo, RBVars)
    get_basis_success = true
  else
    get_basis_success = false
  end

  if !get_snapshots_success && !get_basis_success
    error("Impossible to assemble the reduced problem if
      neither the snapshots nor the bases can be loaded")
  end

  if get_snapshots_success && !get_basis_success
    println("Failed to import the reduced basis, building it via POD")
    assemble_reduced_basis(RBInfo, RBVars)
  end

  if RBInfo.get_offline_structures
    operators = get_offline_structures(RBInfo, RBVars)
    if !isempty(operators)
      assemble_offline_structures(RBInfo, RBVars, operators)
    end
  else
    assemble_offline_structures(RBInfo, RBVars)
  end

  if RBInfo.save_offline
    save_offline(RBInfo, RBVars, operators)
  end

end

################################## ONLINE ######################################

function reconstruct_FEM_solution(RBVars::ROMInfoS)
  println("Reconstructing FEM solution")
  RBVars.x̃ = Broadcasting(*)(RBVars.Φₛ, RBVars.xₙ)
end

function online_phase(
  RBInfo::ROMInfoS,
  RBVars::RBS{T},
  param_nbs) where T

  function get_S_var(var::String, nb::Int)
    load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_snap_path(RBInfo), "$(var)ₕ.csv"))[:, nb]
  end

  function get_norms(solₕ)
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

  function save_online()
    save_CSV(mean_pointwise_err, joinpath(RBInfo.results_path, "mean_point_err.csv"))
    save_CSV(mean_err, joinpath(RBInfo.results_path, "err.csv"))

    times = times_dictionary(RBInfo, RBVars.offline_time, mean_online_time)
    CSV.write(joinpath(RBInfo.results_path, "times.csv"), times)
  end

  function post_process()
    pass_to_pp = Dict("res_path"=>RBInfo.results_path, "FEMSpace"=>FEMSpace,
      "mean_point_err"=>Float.(mean_pointwise_err))
    post_process(RBInfo, pass_to_pp)
  end

  FEMSpace, μ = get_FEMμ_info(RBInfo.FEMInfo)
  get_norm_matrix(RBInfo, RBVars)

  mean_err, mean_pointwise_err, mean_online_time = Vector{T}[], Vector{T}[], 0.
  for nb in param_nbs
    println("Considering parameter number: $nb")

    assemble_RB_system(FEMSpace, RBInfo, RBVars, μ[nb])
    solve_RB_system(RBVars)
    reconstruct_FEM_solution(RBVars)

    mean_online_time += RBVars.online_time / length(param_nbs)

    get_S(var) = get_S_var(var, nb)
    xₕ = Broadcasting(get_S)(RBInfo.vars)
    norms = get_norms(xₕ)
    mean_err, mean_pointwise_err +=
      Broadcasting(compute_errors)(xₕ, RBVars.x̃, RBVars.X₀, norms) / length(param_nbs)

    println("Online wall time (snapshot number $nb): $(RBVars.online_time)s ")
  end

  if RBInfo.save_online
    save_online()
  end

  if RBInfo.post_process
    post_process()
  end;

end
