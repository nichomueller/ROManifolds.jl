include("config_fem.jl")
include("../../../utils/general.jl")
include("../../../ROM/RB_superclasses.jl")
include("../../../ROM/RB_utils.jl")

RB_method = "S-GRB"
training_percentage = 0.8
considered_snaps = convert(Int64, nₛ * training_percentage)
ϵₛ = 1e-5
perform_RHS_DEIM = false
postprocess = false
import_snapshots = true
import_offline_structures = false
save_offline_structures = true
save_results = true

case = 2

if case === 0

  problem_nonlinearities = Dict("Ω" => false, "A" => false, "f" => false, "g" => false, "h" => false)

elseif case === 1

  problem_nonlinearities = Dict("Ω" => false, "A" => true, "f" => false, "g" => false, "h" => false)

elseif case === 2

  problem_nonlinearities = Dict("Ω" => false, "A" => true, "f" => true, "g" => true, "h" => false)

elseif case === 3

  problem_nonlinearities = Dict("Ω" => true, "A" => false, "f" => false, "g" => false, "h" => false)

else

  @error "Only implemented 3 cases for the Poisson steady FEM/ROM simulation"

end

struct ROMSpecifics <: RBProblem
  case::Int64
  paths::Function
  RB_method::String
  problem_nonlinearities::Dict
  nₛ::Int64
  ϵₛ::Float64
  perform_RHS_DEIM::Bool
  postprocess::Bool
  import_snapshots::Bool
  import_offline_structures::Bool
  save_offline_structures::Bool
  save_results::Bool
end
