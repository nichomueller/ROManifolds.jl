include("../../../utils/general.jl")
include("../../../FEM/FEM_superclasses.jl")
include("../../../FEM/FEM_utils.jl")

problem_name = "poisson"
problem_type = "steady"
order = 1
solver = "lu"
nₛ = 100
root = "/home/user1/git_repos/Mabla.jl"
case = 1

if case === 0

  problem_nonlinearities = Dict("Ω" => false, "A" => false, "f" => false, "g" => false, "h" => false)
  ranges = [0., 1.]
  mesh_name = "model.json"
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_nonlinearities)
  dirichlet_tags = ["diri0", "dirig"]
  neumann_tags = []
  dirichlet_bnds = [["sides", "sides_c"], ["circle","circle_c", "triangle", "triangle_c", "square", "square_c"]]
  neumann_bnds = []

elseif case === 1

  problem_nonlinearities = Dict("Ω" => false, "A" => true, "f" => false, "g" => false, "h" => false)
  ranges = [0., 1.]
  mesh_name = "model.json"
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_nonlinearities)
  dirichlet_tags = ["diri0", "dirig"]
  neumann_tags = []
  dirichlet_bnds = [["sides", "sides_c"], ["circle","circle_c", "triangle", "triangle_c", "square", "square_c"]]
  neumann_bnds = []

elseif case === 2

  problem_nonlinearities = Dict("Ω" => false, "A" => true, "f" => true, "g" => true, "h" => false)
  ranges = [0., 1.]
  mesh_name = "model.json"
  paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_nonlinearities)
  dirichlet_tags = ["diri0", "dirig"]
  neumann_tags = []
  dirichlet_bnds = [["sides", "sides_c"], ["circle","circle_c", "triangle", "triangle_c", "square", "square_c"]]
  neumann_bnds = []

else

  @error "Only implemented 3 cases for the Poisson steady FEM simulation"

end

problem_info = ProblemSpecifics(case, probl_nl, order, dirichlet_tags, dirichlet_bnds, neumann_tags, neumann_bnds, solver, paths)
