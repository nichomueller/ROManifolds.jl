module RBTransient

using LinearAlgebra
using BlockArrays
using SparseArrays
using DrWatson
using Kronecker
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

using Mabla.FEM
using Mabla.FEM.Utils
using Mabla.FEM.IndexMaps
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady

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

export get_stage_operator
include("ThetaMethod.jl")

include("PostProcess.jl")

end # module
