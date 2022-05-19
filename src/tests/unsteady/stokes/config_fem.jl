include("../../../utils/general.jl")
include("../../../FEM/FEM_superclasses.jl")
include("../../../FEM/FEM_utils.jl")
include("../../../FEM/FESpaces.jl")
include("../../../FEM/assemblers.jl")
include("../../../FEM/solvers.jl")

problem_name = "stokes"
problem_ntuple = (0,0)
problem_type = "unsteady"
problem_dim = 3
order = 1
solver = "lu"
time_method = "θ-method"
θ = 0.5
RK_type = :SDIRK_2_1_2
t₀ = 0.
T = 5 / 4
δt = 0.025
nₛ = 100
root = "/home/user1/git_repos/Mabla.jl"
case = 1

if case === 0

  probl_nl = Dict("Ω" => false, "M" => false, "A" => false, "f" => false, "g" => false, "h" => false)
  ranges = [[1.,2.] [4., 8.] [0.1, 0.3]]
  mesh_name = "tube1x2_h0.15.msh"
  dirichlet_tags = ["wall","inlet"]
  neumann_tags = ["outlet"]
  dirichlet_labels = [[1] [2]]
  neumann_labels = [[3]]

elseif case === 1

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => false, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1] [4., 8.] [0.1, 0.3]]
  mesh_name = "tube1x2_h0.15.msh"
  dirichlet_tags = ["wall","inlet"]
  neumann_tags = ["outlet"]
  dirichlet_labels = [[1] [2]]
  neumann_labels = [[3]]

elseif case === 2

  probl_nl = Dict("Ω" => false, "M" => false, "A" => false, "f" => true, "g" => false, "h" => false)
  ranges = [[1.,2.] [0.1, 0.3] [4., 8.] [0.1, 0.3] [0.2, 0.8]]
  mesh_name = "bypass.msh"
  dirichlet_tags = ["wall","inlet1","inlet2"]
  neumann_tags = ["outlet"]
  dirichlet_labels = [[1] [2] [3]]
  neumann_labels = [[4]]

elseif case === 3

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => true, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1] [4., 8.] [0.1, 0.3] [0.2, 0.8]]
  mesh_name = "bypass.msh"
  dirichlet_tags = ["wall","inlet1","inlet2"]
  neumann_tags = ["outlet"]
  dirichlet_labels = [[1] [2] [3]]
  neumann_labels = [[4]]

else

  @error "Only implemented 4 cases for the Stokes unsteady FEM simulation"

end

paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, case)
problem_info = ProblemSpecificsUnsteady(case, probl_nl, order, dirichlet_tags, dirichlet_labels, neumann_tags, neumann_labels, solver, paths, time_method, θ, RK_type, t₀, T, δt)
model = DiscreteModelFromFile(paths.mesh_path)
FE_space = get_FESpace(problem_ntuple, problem_info, model)
FE_space₀ = get_FESpace(problem_ntuple, problem_info, model)
