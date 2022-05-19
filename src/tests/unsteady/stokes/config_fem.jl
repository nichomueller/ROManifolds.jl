include("../../../utils/general.jl")
include("../../../FEM/FEM_superclasses.jl")
include("../../../FEM/FEM_utils.jl")

problem_name = "poisson"
problem_type = "unsteady"
problem_dim = 3
order = 1
solver = "lu"
time_method = "θ-method"
θ = 0.5
RK_type = :SDIRK_2_1_2
t₀ = 0.
T = 10 / 4
δt = 0.05
nₛ = 100
root = "/home/user1/git_repos/Mabla.jl"
case = 1

if case === 0

  probl_nl = Dict("Ω" => false, "M" => false, "A" => false, "f" => false, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]]
  mesh_name = "tube1x2_h0.15.msh"
  dirichlet_tags = ["wall"]
  neumann_tags = ["outlet"]
  dirichlet_labels = []
  neumann_labels = []

elseif case === 1

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => false, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1]]
  mesh_name = "tube1x2_h0.15.msh"
  dirichlet_tags = ["wall"]
  neumann_tags = ["outlet"]
  dirichlet_labels = []
  neumann_labels = []

elseif case === 2

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => true, "g" => false, "h" => false)
  ranges = [[0.4, 0.6] [0.4, 0.6] [0.05, 0.1] [0.0, 1.0] [0.0, 1.0]]
  mesh_name = "bypass.msh"
  dirichlet_tags = ["wall"]
  neumann_tags = ["outlet"]
  dirichlet_labels = []
  neumann_labels = []

else

  @error "Only implemented 3 cases for the Poisson steady FEM simulation"

end

paths = FEM_paths(root, problem_type, problem_name, mesh_name, problem_dim, case)
problem_info = ProblemSpecificsUnsteady(case, probl_nl, order, dirichlet_tags, dirichlet_labels, neumann_tags, neumann_labels, solver, paths, time_method, θ, RK_type, t₀, T, δt)
