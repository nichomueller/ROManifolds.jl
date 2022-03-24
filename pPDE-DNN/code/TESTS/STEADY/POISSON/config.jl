include("../../../UTILS/general.jl")

const problem_type = "STEADY"  # STEADY / UNSTEADY
const problem_name = "POISSON"   # POISSON / STOKES / NAVIER-STOKES / BURGERS
const approx_type = "ROM"   # ROM / DL
const problem_dim = 3   
const problem_nonlinearities = Dict("A" => false,  # true / false
                                    "f" => false,  # true / false
                                    "g" => false,  # true / false
                                    "h" => false)  # true / false
const number_coupled_blocks = 1
const mesh_name = "model"
const use_lifting = false

@assert !(problem_name === "BURGERS" && problem_type === "STEADY")

const root = "/home/user1/git_repos/pPDE-NN/code"

function FEM_ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim)
    mesh_path = joinpath(root, joinpath("FEM", joinpath("models", mesh_name)))
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
    return mesh_path, FEM_snap_path, FEM_structures_path, FOM_files_extension, FOM_files_delimiter, basis_path, ROM_structures_path, gen_coords_path,       results_path
end

struct probl_specifics
    paths::String
    problem_type::String
    problem_name::String
    approx_type::String
    problem_dim::Int
    problem_nonlinearities::Dict
    number_coupled_blocks::Int# true / false
    #time_marching_scheme::String
end


struct ROM_specifics# true / false
    RB_method::String
    n_snaps::Int64
    ϵˢ_POD::Float64
    #ϵᵗ_POD::Float64
    preprocess::Bool
    postprocess::Bool
    train_percentage::Float64
    import_snapshots::Bool
    #dir_dofs_ids = get_dirichlet_dof_ids(FE_space.V₀)
    #g_on_dir_bnd = g()
    #interpolate_dirichlet(g, FE_space.V₀)
    #gₕ_at_Qₕ = interpolate_dirichlet(gₕ, FE_space.Qₕ_cell_point)
    import_offline_structures::Bool
    save_offline_structures::Bool
    save_results::Bool
end

if problem_nonlinearities["f"] === false
    f(x, μ) = 1.0
else
    f(x, μ) = sin(μ .* x)
end

if problem_nonlinearities["g"] === false
    g(x, μ) = 2.0
else
    g(x, μ) = sin(μ .* x)
end

if problem_nonlinearities["h"] === false
    h(x, μ) = 3.0
else
    h(x, μ) = sin(μ .* x)
end

if problem_nonlinearities["A"] === false
    const σ = Point(rand(Uniform(0.4, 0.6)), rand(Uniform(0.4, 0.6)), rand(Uniform(0.05, 0.1)))
    μ(x) = σ[3] + 1 / σ[3] * exp(-((x[1] - σ[1])^2 + (x[2] - σ[2])^2) / σ[3])
else
    μ = rand(Uniform(0.01, 10.0))
end

all_paths = FEM_ROM_paths(root, problem_type, problem_name, mesh_name, problem_dim)
probl_info = probl_specifics(all_paths, problem_type, problem_name, approx_type, problem_dim, problem_nonlinearities, number_coupled_blocks)
ROM_info = ROM_specifics("S-RB", 10, 1e-5, false, false, 0.8, true, true, true, true)





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
    create_dir(ROM_path)OM_info.lin_solver
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