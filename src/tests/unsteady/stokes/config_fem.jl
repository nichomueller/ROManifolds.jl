include("../../../utils/general.jl")
include("../../../FEM/FEM_superclasses.jl")
include("../../../FEM/FEM_utils.jl")
include("../../../FEM/FESpaces.jl")
include("../../../FEM/assemblers.jl")
include("../../../FEM/solvers.jl")

problem_name = "stokes"
problem_ntuple = (0,0)
problem_type = "unsteady"
order = 2
solver = "lu"
time_method = "θ-method"
θ = 0.5
RK_type = :SDIRK_2_1_2
t₀ = 0.
T = 5 / 4
δt = 0.025
nₛ = 100
root = "/home/user1/git_repos/Mabla.jl"
case = 0

if case == 0

  probl_nl = Dict("Ω" => false, "M" => false, "A" => false, "f" => false, "g" => true, "h" => false)
  ranges = [[1.,2.] [4., 8.] [0.1, 0.3]]
  mesh_name = "my_cube.json"
  dirichlet_tags = ["dir0","dir1"]
  neumann_tags = ["neum"]
  dirichlet_bnds = [collect(1:13), collect(14:26)]
  neumann_bnds = [[28]]

elseif case == 1

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => false, "g" => true, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1] [4., 8.] [0.1, 0.3]]
  mesh_name = "tube.json"
  dirichlet_tags = ["wall","inlet"]
  neumann_tags = ["outlet"]
  dirichlet_bnds = [[6] [1,2,3,4,5,7,8]]
  neumann_bnds = []

elseif case == 2

  probl_nl = Dict("Ω" => false, "M" => false, "A" => false, "f" => true, "g" => true, "h" => false)
  ranges = [[1.,2.] [0.1, 0.3] [4., 8.] [0.1, 0.3] [0.2, 0.8]]
  mesh_name = "bypass_coarse_fluid.mesh"
  dirichlet_tags = ["wall","inlet1","inlet2"]
  neumann_tags = ["outlet"]
  dirichlet_bnds = []
  neumann_bnds = []

elseif case == 3

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => true, "g" => true, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1] [4., 8.] [0.1, 0.3] [0.2, 0.8]]
  mesh_name = "bypass_coarse_fluid.mesh"
  dirichlet_tags = ["wall","inlet1","inlet2"]
  neumann_tags = ["outlet"]
  dirichlet_bnds = []
  neumann_bnds = []

else

  @error "Only implemented 4 cases for the Stokes unsteady FEM simulation"

end

paths = FEM_paths(root, problem_type, problem_name, mesh_name, case)
const problem_info = ProblemSpecificsUnsteady(case, probl_nl, order, dirichlet_tags,
  dirichlet_bnds, neumann_tags, neumann_bnds, solver, paths, time_method, θ,
  RK_type, t₀, T, δt)
const model = DiscreteModelFromFile(paths.mesh_path)
const FE_space = get_FESpace(problem_ntuple, problem_info, model)
const FE_space₀ = get_FESpace(problem_ntuple, problem_info, model)
