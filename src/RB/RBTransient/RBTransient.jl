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

using ROM.Utils
using ROM.DofMaps
using ROM.TProduct
using ROM.ParamDataStructures
using ROM.ParamAlgebra
using ROM.ParamFESpaces
using ROM.ParamSteady
using ROM.ParamODEs

using ROM.RBSteady

import Base: +,-,*,\
import UnPack: @unpack
import Gridap.Algebra: allocate_matrix,allocate_vector,solve
import PartitionedArrays: tuple_of_arrays
import ROM.RBSteady: _get_label

export TransientReduction
export TransientMDEIMReduction
include("ReductionMethods.jl")

export TransientSnapshots
export TransientGenericSnapshots
export TransientSparseSnapshots
export TransientSnapshotsWithIC
export UnfoldingTransientSnapshots
export ModeTransientSnapshots
include("Snapshots.jl")

include("RBSolver.jl")

include("TTLinearAlgebra.jl")

include("BasisConstruction.jl")

include("GalerkinProjections.jl")

export TransientProjection
include("Projections.jl")

include("RBSpace.jl")

export TransientMDEIM
include("HyperReduction.jl")

include("HRParamArray.jl")

export TransientRBOperator
export GenericTransientRBOperator
export LinearNonlinearTransientRBOperator
include("ReducedOperators.jl")

include("ThetaMethod.jl")

include("PostProcess.jl")

end # module
