module RBTransient

using LinearAlgebra
using BlockArrays
using SparseArrays
using DrWatson
using Serialization

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs
using Gridap.TensorValues
using Gridap.Helpers

using ReducedOrderModels.Utils
using ReducedOrderModels.IndexMaps
using ReducedOrderModels.TProduct
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamAlgebra
using ReducedOrderModels.ParamFESpaces
using ReducedOrderModels.ParamSteady
using ReducedOrderModels.ParamODEs

using ReducedOrderModels.RBSteady

import Base: +,-,*,\
import UnPack: @unpack
import Gridap.Algebra: allocate_matrix,allocate_vector,solve
import PartitionedArrays: tuple_of_arrays

export TransientReduction
export TransientMDEIMReduction
include("ReductionMethods.jl")

export AbstractTransientSnapshots
export TransientGenericSnapshots
export TransientSparseSnapshots
export UnfoldingTransientSnapshots
export ModeTransientSnapshots
include("Snapshots.jl")

include("RBSolver.jl")

include("TTLinearAlgebra.jl")

include("BasisConstruction.jl")

include("GalerkinProjections.jl")

export TransientBasis
include("Projections.jl")

export TransientRBSpace
export TransientEvalMultiFieldRBSpace
include("RBSpace.jl")

export TransientMDEIM
export TransientAffineDecomposition
include("HyperReduction.jl")

export TransientRBOperator
export GenericTransientRBOperator
export LinearNonlinearTransientRBOperator
include("ReducedOperators.jl")

export TransientRBNewtonRaphsonOp
include("RBNewtonRaphsonOperator.jl")

include("ThetaMethod.jl")

include("PostProcess.jl")

end # module
