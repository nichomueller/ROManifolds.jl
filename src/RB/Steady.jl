################################# OFFLINE ######################################
function get_snapshot_matrix(
  RBInfo::ROMInfoS,
  RBVars::RBProblemS{T}) where T

  function get_S_var(var::String)
    println("Importing the snapshot matrix for field $var,
      number of snapshots considered: $(RBInfo.nₛ)")
    S = load_CSV(Matrix{T}(undef,0,0),
      joinpath(get_FEM_snap_path(RBInfo), "$(var)ₕ.csv"))[:, 1:RBInfo.nₛ]
    println("Dimension of snapshot matrix: $(size(S))")

    S, size(S)[1]
  end

  RBVars.S, RBVars.Nₛ = Broadcasting(get_S_var)(RBInfo.problem_unknowns);

end

function assemble_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::RBProblemS)

  assemble_reduced_basis_space(RBInfo, RBVars);

end

function get_reduced_basis(
  RBInfo::ROMInfoS,
  RBVars::RBProblemS)

  get_reduced_basis_space(RBInfo, RBVars);

end

function get_offline_structures(
  RBInfo::ROMInfoS,
  RBVars::RBProblemS{T}) where T

  function get_Var(Var::MVVariable)

    var = Var.var

    if !(var ∈ RBInfo.FEM_vecs && RBInfo.online_RHS)
      if isfile(joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))
        Var.Matₙ = load_CSV(Matrix{T}[],
          joinpath(RBInfo.ROM_structures_path, "$(var)ₙ.csv"))
        if var ∈ RBInfo.probl_nl
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

function save_assembled_structures(
  RBInfo::ROMInfoS,
  RBVars::RBProblemS{T},
  operators::Vector{String}) where T

  save_CSV(RBVars.Φₛ, joinpath(RBInfo.ROM_structures_path,"Φₛ.csv"))

  save_Var(Var) = save_Var_structures(Var, operators)
  Broadcasting(save_Var)(RBVars.Vars);

end

function offline_phase(
  RBInfo::ROMInfoS,
  RBVars::RBProblemS)

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

  if RBInfo.save_offline_structures
    save_assembled_structures(RBInfo, RBVars, operators)
  end

end
