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
using GridapSolvers.LinearSolvers, GridapSolvers.NonlinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, NonlinearSystemBlock, BiformBlock, BlockTriangularSolver

using Mabla.FEM.ParamDataStructures

import UnPack: @unpack
import ArraysOfArrays: innersize
import PartitionedArrays: tuple_of_arrays

export ParamCounter
export ParamInserterCSC
export eltype2
include("ParamAlgebraInterfaces.jl")

include("ParamIterativeSolvers.jl")

end # module
