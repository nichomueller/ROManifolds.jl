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

using GridapEmbedded
using GridapEmbedded.Interfaces

export PerformanceTracker
export CostTracker
export SU
export reset_tracker!
export update_tracker!
export compute_speedup
export compute_error
export compute_relative_error
export induced_norm
include("PerformanceTrackers.jl")

export PartialFunctions
export PartialDerivative
export PartialTrace
export ∂ₓ₁, ∂ₓ₂, ∂ₓ₃
include("PartialFunctions.jl")

export get_values
export get_parent
export order_triangulations
export find_closest_view
include("Triangulations.jl")

export Contribution
export ArrayContribution
export VectorContribution
export MatrixContribution
export TupOfArrayContribution
export contribution
export change_domains
include("Contributions.jl")

end # module
