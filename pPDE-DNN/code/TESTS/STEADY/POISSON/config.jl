include("config_fom.jl")

const RB_method = "S-GRB"
const ϵˢ = 1e-5
const training_percentage = 0.8
const considered_snaps = convert(Int64, nₛ * training_percentage)
#const ϵᵗ_POD::Float64
const preprocess = false
const postprocess = false
const import_snapshots = true
const import_offline_structures = false
const save_offline_structures = true
const save_results = true

function ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim, RB_method)
    paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)  
    ROM_path = joinpath(path.cur_mesh_path, RB_method)
    create_dir(ROM_path)
    basis_path = joinpath(ROM_path, "basis")
    create_dir(basis_path)
    ROM_structures_path = joinpath(ROM_path, "ROM_structures")
    create_dir(ROM_structures_path)
    gen_coords_path = joinpath(ROM_path, "gen_coords")
    create_dir(gen_coords_path) 
    results_path = joinpath(ROM_path, "results")
    create_dir(results_path)
    return paths.FEM_snap_path, paths.FEM_structures_path, basis_path, ROM_structures_path, gen_coords_path, results_path
end


struct ROM_specifics
    paths::Function
    RB_method::String
    problem_nonlinearities::Dict
    nₛ::Int64
    ϵˢ::Float64
    #ϵᵗ_POD::Float64
    preprocess::Bool
    postprocess::Bool
    train_percentage::Float64
    import_snapshots::Bool
    import_offline_structures::Bool
    save_offline_structures::Bool
    save_results::Bool
end




