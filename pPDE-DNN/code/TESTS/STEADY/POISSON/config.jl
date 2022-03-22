include("../../../UTILS/general.jl")

const problem_type = "STEADY"  # STEADY / UNSTEADY
const problem_name = "POISSON"   # POISSON / STOKES / NAVIER-STOKES / BURGERS
const approx_type = "ROM"   # ROM / DL
const problem_dim = 3   
const problem_nonlinearities = Dict("A" => false,  # true / false
                                    "f" => false)  # true / false
const number_coupled_blocks = 1

@assert !(problem_name === "POISSON" && problem_type === "UNSTEADY") || (problem_name === "BURGERS" && problem_type === "STEADY")

const root = "/home/user1/git_repos/pPDE-NN/code"
const mesh_name = "geometry_name"

function FEM_ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim)
    root_test = joinpath(root, joinpath("TESTS", problem_type * "_" * problem_name * "_" * problem_dim))
    FEM_path = joinpath(root_test, joinpath(mesh_name, "FEM_data"))
    FEM_snap_path = joinpath(FEM_path, "snapshots")
    FEM_structures_path = joinpath(FEM_path, "FEM_structures")
    FOM_files_extension = "txt" 
    FOM_files_delimiter = ","
    name_test = ROM_specifics["reduction_method"]
    ROM_path = joinpath(root_test, joinpath(mesh_name, name_test))
    create_dir(ROM_path)
    basis_path = joinpath(ROM_path, "basis")
    create_dir(basis_path)
    ROM_structures_path = joinpath(ROM_path, "ROM_structures")
    create_dir(ROM_structures_path)
    gen_coords_path = joinpath(ROM_path, "gen_coords")
    create_dir(gen_coords_path) 
    results_path = joinpath(ROM_path, "results")
    create_dir(results_path)
    (paths) -> (FEM_snap_path; FEM_structures_path; FOM_files_extension; FOM_files_delimiter; basis_path; ROM_structures_path; gen_coords_path; results_path)
end

struct FOM_specifics
    Nᵤˢ::Int64
    #time_dimension_FOM::Int64
    #inal_time::Float64
    #time_marching_scheme::String
end

struct ROM_specifics
    RB_method::String
    n_snaps::Int64
    ϵˢ_POD::Float64
    #ϵᵗ_POD::Float64
    preprocess::Bool
    postprocess::Bool
    train_percentage::Float64
    import_snapshots::Bool
    import_offline_structures::Bool
    save_offline_structures::Bool
    save_results::Bool
end

FOM_info = FOM_specifics(40000, 120, 0.3, "BDF2")
ROM_info = ROM_specifics("S-RB", 10, 1e-5, 1e-5, false, false, 0.8, true, true, true, true)





#=

struct FEM_ROM_paths
    FEM_path
    FEM_snap_path
    FEM_structures_path
    DEIM_MDEIM_path
    ROM_path
    basis_path
    ROM_structures_path
    gen_coords_path
    results_path
end

FOM_specifics = Dict("mesh_name" => mesh_name,
                     "model" => problem_type * "/" * problem_name,
                     "dimension" => 3,
                     "space_dimension_FOM" => [40000],
                     "time_dimension_FOM" => 100,
                     "final_time" => 1,
                     "time_marching_scheme" => "BDF2",
                     "number_coupled_blocks" => 1)

ROM_specifics = Dict("reduction_method" => "ST-GRB",
                     "n_snapshots" => 10,
                     "ϵ_POD_space" => 1e-5,
                     "ϵ_POD_time" => 1e-5,
                     "preprocess" => false,
                     "postprocess" => false,
                     "train_percentage" => 0.8,
                     "import_snapshots" => true,
                     "import_offline_structures" => true,
                     "save_offline_structures" => true,
                     "save_results" => true)

DL_specifics = Dict("architecture" => "---",
                    "weight_initialization" => "Xavier",
                    "pretrain" => false,
                    "ϵ_POD_time" => 1e-5,
                    "preprocess" => true,
                    "postprocess" => false,
                    "activation" => true,
                    "optimizer" => true,
                    "save_results" => true)

FEM_path = joinpath(root_test, joinpath(mesh_name, "FEM_data"))
FEM_snap_path = joinpath(FEM_path, "snapshots")
FEM_structures_path = joinpath(FEM_path, "FEM_structures")
FOM_files_extension = "txt" 
FOM_files_delimiter = ","

if problem_nonlinearities["A"] === true || problem_nonlinearities["A"] === true
    DEIM_MDEIM_path = joinpath(FEM_structures_path, "DEIM_MDEIM")
end

if approx_type === "ROM"
    name_test = ROM_specifics["reduction_method"]
    ROM_path = joinpath(root_test, joinpath(mesh_name, name_test))
    create_dir(ROM_path)
    basis_path = joinpath(ROM_path, "basis")
    create_dir(basis_path)
    ROM_structures_path = joinpath(ROM_path, "ROM_structures")
    create_dir(ROM_structures_path)
    gen_coords_path = joinpath(ROM_path, "gen_coords")

else
    @error "Error: unrecognized approximation method - must choose between 'ROM' and 'DL' "
    throw(ArgumentError)
end


results_path = joinpath(ROM_path, "results")
create_dir(results_path)

struct DL_specifics
    architecture
    weight_initialization
    pretrain
    ϵ_POD_time
    preprocess
    postprocess
    activation
    optimizer
    save_results
end

function f(b::FOM_specifics)
    println(b.mesh_name)
end
=#