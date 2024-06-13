module RBSteady

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

import Base: +,-,*,\
import UnPack: @unpack
import Gridap.Algebra: allocate_matrix,allocate_vector,solve
import PartitionedArrays: tuple_of_arrays

export tpod
export ttsvd
export gram_schmidt!
export orth_complement!
export orth_projection
include("BasisConstruction.jl")

export AbstractSnapshots
export BasicSnapshots
export SnapshotsAtIndices
export SparseSnapshots
export StandardSnapshots
export StandardSteadySnapshots
export BlockSnapshots
export Snapshots
export flatten_snapshots
export select_snapshots
export select_snapshots_entries
export get_touched_blocks
export get_realization
include("Snapshots.jl")

export AbstractTTCore
export TTCore
export SparseCore
export SparseCoreCSC
include("TTCores.jl")

export RBSolver
export MDEIMStyle
export SpaceMDEIM
export get_test_directory
export fe_solutions
export nonlinear_rb_solve!
include("RBSolver.jl")

export Projection
export SteadyProjection
export PODBasis
export TTSVDCores
export BlockProjection
export get_basis_space
export num_fe_dofs
export num_reduced_dofs
export get_cores
export get_spatial_cores
export cores2basis
export enrich_basis
export add_space_supremizers
include("Projections.jl")

export RBSpace
export BlockRBSpace
export reduced_fe_space
export reduced_basis
export pod_error
include("RBSpace.jl")

export RBOperator
export PODOperator
export reduced_operator
export get_fe_trial
export get_fe_test
export jacobian_and_residual
include("PODOperator.jl")

export ReducedAlgebraicOperator
export ReducedVectorOperator
export ReducedMatrixOperator
export reduce_operator
export compress_cores
export compress_core
export multiply_cores
include("ReducedAlgebraicOperator.jl")

export AbstractIntegrationDomain
export AffineDecomposition
export AffineContribution
export BlockAffineDecomposition
export mdeim
export get_mdeim_indices
export get_integration_domain
export reduce_triangulation
export reduced_jacobian
export reduced_residual
export reduced_jacobian_residual
export allocate_coefficient
export allocate_result
export coefficient!
export mdeim_result
include("AffineDecomposition.jl")

export PODMDEIMOperator
export LinearNonlinearPODMDEIMOperator
export fe_jacobian!
export fe_residual!
export pod_mdeim_error
include("PODMDEIMOperator.jl")

export ComputationalStats
export RBResults
export rb_results
export load_solve
export compute_speedup
export compute_error
include("PostProcess.jl")

end # module
