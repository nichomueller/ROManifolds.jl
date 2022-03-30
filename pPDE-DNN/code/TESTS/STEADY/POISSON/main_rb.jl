Pkg.activate(".")
include("config.jl")
include("../../../ROM/RB_utils.jl")

paths = FEM_ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim, RB_method)
ROM_info = ROM_specifics(paths, RB_method, problem_nonlinearities, considered_snaps, ϵˢ, preprocess, postprocess, train_percentage, import_snapshots, import_offline_structures, save_offline_structures, save_results)

RB_variables = initialize_RB_variables_struct()
build_RB_approximation(ROM_info, RB_variables)