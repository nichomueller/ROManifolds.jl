include("../../../utils/general.jl")
include("../../../FEM/FEM_superclasses.jl")
include("../../../FEM/FEM_utils.jl")

problem_name = "poisson"
problem_type = "steady"
problem_dim = 3
order = 1
solver = "lu"
nₛ = 100
root = "/home/user1/git_repos/Mabla.jl"
case = 2

if case === 0

  problem_nonlinearities = Dict("Ω" => false, "A" => false, "f" => false, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]]
  mesh_name = "model.json"
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)
  dirichlet_tags = ["sides"]
  neumann_tags = ["circle", "triangle", "square"]

elseif case === 1

  problem_nonlinearities = Dict("Ω" => false, "A" => true, "f" => false, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]]
  mesh_name = "model.json"
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)
  dirichlet_tags = ["sides"]
  neumann_tags = ["circle", "triangle", "square"]

elseif case === 2

  problem_nonlinearities = Dict("Ω" => false, "A" => true, "f" => true, "g" => true, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1] [0.0, 1.0] [0.0, 1.0]]
  mesh_name = "model.json"
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)
  dirichlet_tags = ["sides"]
  neumann_tags = ["circle", "triangle", "square"]

elseif case === 3

  problem_nonlinearities = Dict("Ω" => true, "A" => false, "f" => false, "g" => false, "h" => false)
  ranges = [0.0, 1.0]
  mesh_name = "stretch_model"
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, problem_nonlinearities)
  dirichlet_tags = "boundary"
  neumann_tags = []

else

  @error "Only implemented 3 cases for the Poisson steady FEM simulation"

end

problem_info = ProblemSpecifics(case, order, dirichlet_tags, neumann_tags, solver, paths, problem_nonlinearities)
