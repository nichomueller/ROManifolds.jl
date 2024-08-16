module RBSteady

using LinearAlgebra
using RandomizedLinAlg
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

import Base: +,-,*,\
import UnPack: @unpack
import ArraysOfArrays: innersize
import Gridap.Algebra: allocate_matrix,allocate_vector,solve
import PartitionedArrays: tuple_of_arrays

export AbstractSnapshots
export BasicSnapshots
export SnapshotsAtIndices
export SparseSnapshots
export UnfoldingSteadySnapshots
export BlockSnapshots
export Snapshots
export flatten_snapshots
export select_snapshots
export select_snapshots_entries
export get_touched_blocks
export get_realization
export get_indexed_values
export num_space_dofs
include("Snapshots.jl")

export AbstractTTCore
export SparseCore
export SparseCoreCSC
export BlockCore
export cat_cores
include("TTCores.jl")

export RBSolver
export MDEIMStyle
export SpaceMDEIM
export get_test_directory
export get_fe_solver
export get_fe_stats
export get_rb_offline_stats
export get_rb_online_stats
export fe_solutions
export nonlinear_rb_solve!
include("RBSolver.jl")

export truncated_pod
export ttsvd
export gram_schmidt!
export orth_complement!
export orth_projection
include("BasisConstruction.jl")

export Projection
export SteadyProjection
export PODBasis
export TTSVDCores
export BlockProjection
export get_basis_space
export num_fe_dofs
export num_reduced_dofs
export get_cores
export get_cores_space
export cores2basis
export enrich_basis
export add_space_supremizers
include("Projections.jl")

export FESubspace
export RBSpace
export MultiFieldRBSpace
export fe_subspace
export reduced_fe_space
export reduced_basis
export pod_error
include("RBSpace.jl")

export RBOperator
export PGOperator
export reduced_operator
export get_fe_trial
export get_fe_test
export jacobian_and_residual
export select_fe_quantities_at_indices
include("PGOperator.jl")

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
export empirical_interpolation
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

export PGMDEIMOperator
export LinearNonlinearPGMDEIMOperator
export fe_jacobian!
export fe_residual!
export pod_mdeim_error
include("PGMDEIMOperator.jl")

export RBResults
export rb_results
export load_solve
include("PostProcess.jl")

end # module
