module Utils

using LinearAlgebra
using BlockArrays
using FillArrays

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.Helpers
using Gridap.MultiField
using Gridap.ReferenceFEs
using Gridap.TensorValues

using GridapEmbedded
using GridapEmbedded.Interfaces

using GridapSolvers
using GridapSolvers.PatchBasedSmoothers
import GridapSolvers.PatchBasedSmoothers: inject!, prolongate!

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
export ∂₁, ∂₂, ∂₃
include("PartialFunctions.jl")

export get_values
export get_parent
export order_domains
include("Triangulations.jl")

export Contribution
export ArrayContribution
export VectorContribution
export MatrixContribution
export TupOfArrayContribution
export contribution
include("Contributions.jl")

end # module
