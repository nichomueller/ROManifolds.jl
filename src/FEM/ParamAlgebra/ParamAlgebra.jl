module ParamAlgebra

using LinearAlgebra
using BlockArrays
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Helpers

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers
using GridapSolvers.MultilevelTools

using Mabla.FEM.ParamDataStructures

import UnPack: @unpack
import PartitionedArrays: tuple_of_arrays

export ParamCounter
export ParamInserterCSC
export ParamCSSR
include("ParamAlgebraInterfaces.jl")

export FastLinearSolver
export FastLUSolver
export CholeskySolver
include("FastLinearSolvers.jl")

include("ParamIterativeSolvers.jl")

end # module
