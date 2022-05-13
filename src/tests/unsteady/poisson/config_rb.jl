include("config_fem.jl")
include("../../../utils/general.jl")
include("../../../ROM/RB_superclasses.jl")
include("../../../ROM/RB_utils.jl")

RB_method = "ST-GRB"
time_reduction_technique = "ST-HOSVD"
perform_nested_POD = false
ϵₛ = 1e-5
ϵₜ = 1e-5
training_percentage = 0.8
considered_snaps = convert(Int64, nₛ * training_percentage)
build_parametric_RHS = false
use_norm_X = false
nₛ_MDEIM = min(30, considered_snaps)
nₛ_DEIM = min(10, considered_snaps)
space_time_M_DEIM = true
functional_M_DEIM = false
@assert !(space_time_M_DEIM && functional_M_DEIM) "Choose only one (M)DEIM technique"
postprocess = true
import_snapshots = false
import_offline_structures = true
save_offline_structures = true
save_results = true

case = 1

if case === 0

  probl_nl = Dict("Ω" => false, "M" => false, "A" => false, "f" => false, "g" => false, "h" => false)

elseif case === 1

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => false, "g" => false, "h" => false)

elseif case === 2

  probl_nl = Dict("Ω" => false, "M" => false, "A" => true, "f" => true, "g" => false, "h" => false)

else

  @error "Only implemented 3 cases for the Poisson unsteady FEM/ROM simulation"

end

test_case = ""
if perform_nested_POD
  test_case *= "_nest"
end
if space_time_M_DEIM
  test_case *= "_st"
end
