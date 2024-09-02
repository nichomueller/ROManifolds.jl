module Utils

using LinearAlgebra

using Gridap
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.Helpers
using Gridap.TensorValues

export PerformanceTracker
export CostTracker
export GenericPerformance
export reset_tracker!
export update_tracker!
export get_stats
export compute_speedup
export compute_error
export induced_norm
include("PerformanceTrackers.jl")

export PartialFunctions
export PartialDerivative
export PartialTrace
export ∂ₓ₁, ∂ₓ₂, ∂ₓ₃
include("PartialFunctions.jl")

include("TriangulationView.jl")

end # module
