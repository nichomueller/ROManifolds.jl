include("config_fem.jl")
include("../../../utils/general.jl")
include("../../../ROM/RB_superclasses.jl")
include("../../../ROM/RB_utils.jl")

RB_method = "S-GRB"
training_percentage = 0.8
considered_snaps = convert(Int64, nₛ * training_percentage)
ϵₛ = 1e-5
use_norm_X = false
build_parametric_RHS = false
nₛ_MDEIM = max(35, considered_snaps)
nₛ_DEIM = min(10, considered_snaps)
postprocess = false
import_snapshots = false
import_offline_structures = true
save_offline_structures = true
save_results = true

case = 2

if case === 0

  probl_nl = Dict("Ω" => false, "A" => false, "f" => false, "g" => false, "h" => false)

elseif case === 1

  probl_nl = Dict("Ω" => false, "A" => true, "f" => false, "g" => false, "h" => false)

elseif case === 2

  probl_nl = Dict("Ω" => false, "A" => true, "f" => true, "g" => true, "h" => false)

elseif case === 3

  probl_nl = Dict("Ω" => true, "A" => false, "f" => false, "g" => false, "h" => false)

else

  @error "Only implemented 4 cases for the Poisson steady FEM/ROM simulation"

end
