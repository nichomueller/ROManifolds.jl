module ParamUtils

using Gridap
using Gridap.Arrays
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.Helpers

using Mabla.FEM.ParamDataStructures

include("LagrangianDofBases.jl")

export get_parent
include("TriangulationParents.jl")

export Contribution
export ArrayContribution
export VectorContribution
export MatrixContribution
export TupOfArrayContribution
export contribution
export get_values
include("Contribution.jl")

end # module
