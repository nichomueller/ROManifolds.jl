include("../../../utils/general.jl")
include("../../../FEM/FEM_superclasses.jl")
include("../../../FEM/FEM_utils.jl")
include("../../../FEM/FESpaces.jl")
include("../../../FEM/assemblers.jl")
include("../../../FEM/solvers.jl")

problem_name = "poisson"
problem_ntuple = (0,)
problem_type = "unsteady"
order = 1
solver = "lu"
time_method = "θ-method"
θ = 0.5
RK_type = :SDIRK_2_1_2
t₀ = 0.
T = 2/4
δt = 0.05
nₛ = 100
root = "/home/user1/git_repos/Mabla.jl"
case = 1

if case == 0

  probl_nl = Dict("Ω" => false, "M" => false, "A" => false, "f" => false, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]]
  mesh_name = "model.json"
  dirichlet_tags = ["sides"]
  neumann_tags = ["circle", "triangle", "square"]
  dirichlet_bnds = []
  neumann_bnds = []

elseif case == 1

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => false, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]]
  mesh_name = "my_cube.json"
  dirichlet_tags = ["boundary"]
  neumann_tags = []
  dirichlet_bnds = []
  neumann_bnds = []

elseif case == 2

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => true, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1] [0.0, 1.0] [0.0, 1.0]]
  mesh_name = "model.json"
  dirichlet_tags = ["sides"]
  neumann_tags = ["circle", "triangle", "square"]
  dirichlet_bnds = []
  neumann_bnds = []

elseif case == 3

  probl_nl = Dict("Ω" => true, "M" => false, "A" => false, "f" => false, "g" => false, "h" => false)
  ranges = [0.0, 1.0]
  mesh_name = "stretch_model"
  dirichlet_tags = "boundary"
  neumann_tags = []
  dirichlet_bnds = []
  neumann_bnds = []

else

  @error "Only implemented 3 cases for the Poisson steady FEM simulation"

end

paths = FEM_paths(root, problem_type, problem_name, mesh_name, case)
const problem_info = ProblemSpecificsUnsteady(case,probl_nl,order,dirichlet_tags,
  dirichlet_bnds,neumann_tags,neumann_bnds,solver,paths,time_method,θ,RK_type,t₀,
  T,δt)
const model = DiscreteModelFromFile(paths.mesh_path)
const FE_space = get_FESpace(problem_ntuple, problem_info, model)
const FE_space₀ = get_FESpace(problem_ntuple, problem_info, model)
