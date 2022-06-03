include("config_rb.jl")

function setup_RB()

  paths = ROM_paths(root, problem_type, problem_name, mesh_name, RB_method, case)
  ROM_info = ROMSpecificsSteady(
    probl_nl,
    case,
    paths,
    RB_method,
    considered_snaps,
    ϵₛ,
    use_norm_X,
    build_parametric_RHS,
    nₛ_MDEIM,
    nₛ_DEIM,
    postprocess,
    import_snapshots,
    import_offline_structures,
    save_offline_structures,
    save_results)
  if RB_method == "S-GRB"
    RB_variables = setup((0,))
  else
    RB_variables = setup((0,0))
  end

  ROM_info, RB_variables

end

function run_RB()
  @info "Setting up the RB solver, steady Poisson problem"
  ROM_info, RB_variables = setup_RB()

  @info "Offline phase of the RB solver, steady Poisson problem"
  build_RB_approximation(ROM_info, RB_variables)

  @info "Online phase of the RB solver, steady Poisson problem"
  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  param_nbs = 95:100
  testing_phase(ROM_info, RB_variables, μ, param_nbs)

end

run_RB()
