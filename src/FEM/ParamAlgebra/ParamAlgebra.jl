module ParamAlgebra

using LinearAlgebra
using BlockArrays
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using Gridap.ReferenceFEs

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers

using ROM.ParamDataStructures

import ArraysOfArrays: innersize
import ROM.DofMaps: OIdsToIds, add_ordered_entries!

include("ParamAlgebraInterfaces.jl")

include("ParamSolvers.jl")

include("ParamIterativeSolvers.jl")

end # module
