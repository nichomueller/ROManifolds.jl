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
using Mabla.FEM.ParamSteady
using Mabla.FEM.ParamODEs

using Mabla.RB
using Mabla.RB.RBSteady

import Base: +,-,*,\
import UnPack: @unpack
import Gridap.Algebra: allocate_matrix,allocate_vector,solve
import PartitionedArrays: tuple_of_arrays

export AbstractTransientSnapshots
export TransientBasicSnapshots
export TransientSnapshots
export TransientSparseSnapshots
export UnfoldingTransientSnapshots
export ModeTransientSnapshots
export compress
include("Snapshots.jl")

export get_stage_operator
include("ThetaMethod.jl")

export SpaceOnlyMDEIM
export SpaceTimeMDEIM
export ThetaMethodRBSolver
include("RBSolver.jl")

include("BasisConstruction.jl")

export TransientProjection
export TransientPODBasis
export TransientTTSVDCores
export get_basis_time
export add_time_supremizers
include("Projections.jl")

export TransientRBSpace
include("RBSpace.jl")

export TransientRBOperator
export TransientPODOperator
include("PODOperator.jl")

export combine_basis_time
include("ReducedAlgebraicOperator.jl")

export TransientIntegrationDomain
export TransientAffineDecomposition
include("AffineDecomposition.jl")

export TransientPODMDEIMOperator
export LinearNonlinearTransientPODMDEIMOperator
include("PODMDEIMOperator.jl")

include("PostProcess.jl")

end # module
