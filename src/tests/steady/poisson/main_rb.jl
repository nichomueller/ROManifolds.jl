include("config_rb.jl")
#include("main_fem.jl")
include("../../../ROM/RB_Poisson_steady.jl")

function setup_RB()

  paths = ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim, RB_method)
  ROM_info = ROM_specifics(case, paths, RB_method, problem_nonlinearities, considered_snaps, ϵₛ, postprocess, import_snapshots, import_offline_structures, save_offline_structures, save_results)
  if RB_method === "S-GRB"
    RB_variables = setup(PoissonSTGRB([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []))
  else
    RB_variables = setup(PoissonSTPGRB([], []))
  end

  ROM_info, RB_variables

end

function run_RB()
  @info "Setting up the RB solver, steady Poisson problem"
  ROM_info, RB_variables = setup_RB()

  @info "Offline phase of the RB solver, steady Poisson problem"
  μ = load_CSV(joinpath(ROM_info.paths.FEM_snap_path, "μ.csv"))
  build_RB_approximation(ROM_info, RB_variables; μ)

  @info "Online phase of the RB solver, steady Poisson problem"
  param_nbs = 15:20
  testing_phase(ROM_info, RB_variables, μ, param_nbs)

end

run_RB()
