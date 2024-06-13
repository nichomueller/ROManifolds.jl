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
using Gridap.Helpers

using Mabla.FEM
using Mabla.FEM.IndexMaps
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamODEs
using Mabla.FEM.ParamTensorProduct
using Mabla.FEM.ParamUtils

using Mabla.RB
using Mabla.RB.RBSteady

import Base: +,-,*,\
import UnPack: @unpack
import Gridap.Algebra: allocate_matrix,allocate_vector,solve
import PartitionedArrays: tuple_of_arrays

include("BasisConstruction.jl")

export AbstractTransientSnapshots
export BasicTransientSnapshots
export TransientSnapshots
export TransientSparseSnapshots
export StandardTransientSnapshots
export ModeTransientSnapshots
export compress
include("Snapshots.jl")

export get_stage_operator
include("ThetaMethod.jl")

export SpaceOnlyMDEIM
export SpaceTimeMDEIM
export ThetaMethodRBSolver
include("RBSolver.jl")

export TransientProjection
export TransientPODBasis
export TransientTTSVDCores
export get_basis_time
export add_time_supremizers
include("Projections.jl")

export TransientRBBasis
include("RBSpace.jl")

include("PODOperator.jl")

export combine_basis_time
include("ReducedAlgebraicOperator.jl")

export TransientIntegrationDomain
export TransientAffineDecomposition
include("AffineDecomposition.jl")

include("PODMDEIMOperator.jl")

include("PostProcess.jl")

end # module
