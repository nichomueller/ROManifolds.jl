include("config_rb.jl")
include("../../../ROM/RB_utils.jl")

paths = FEM_ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim, RB_method)
ROM_info = ROM_specifics(paths, RB_method, problem_nonlinearities, considered_snaps, ϵˢ, preprocess, postprocess, train_percentage, import_snapshots, import_offline_structures, save_offline_structures, save_results)

if RB_method === "S-GRB"
    include("../../../ROM/S-GRB_Poisson.jl")
end

@info "Offline phase, Poisson problem"
Poisson_info = Poisson_RB([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
RB_variables = setup(Poisson_info)
build_RB_approximation(ROM_info, RB_variables)

@info "Online phase, Poisson problem"
param_nbs = 15:20
testing_phase(ROM_info, RB_variables, param_nbs)
