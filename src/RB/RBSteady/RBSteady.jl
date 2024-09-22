module RBSteady

using LinearAlgebra
using LowRankApprox
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

export ReductionStyle
export SearchSVDRank
export FixedSVDRank
export LRApproxRank
export TTSVDRanks
export NormStyle
export EuclideanNorm
export EnergyNorm
export AbstractReduction
export DirectReduction
export GreedyReduction
export AbstractAffineReduction
export AffineReduction
export PODReduction
export TTSVDReduction
export SupremizerReduction
export AbstractMDEIMReduction
export MDEIMReduction
export AdaptiveReduction
export get_reduction
include("ReductionMethods.jl")

export AbstractSnapshots
export GenericSnapshots
export SnapshotsAtIndices
export SparseSnapshots
export UnfoldingSteadySnapshots
export BlockSnapshots
export Snapshots
export flatten_snapshots
export select_snapshots
export get_realization
export get_indexed_values
export num_space_dofs
include("Snapshots.jl")

export AbstractTTCore
export SparseCore
export SparseCoreCSC
include("TTCores.jl")

export TTContraction
export contraction
export sequential_product
export reduced_cores
export cores2basis
include("TTLinearAlgebra.jl")

export RBSolver
export RBOnlineCache
export get_fe_solver
export solution_snapshots
export nonlinear_rb_solve!
include("RBSolver.jl")

export reduction
export tpod
export ttsvd
export gram_schmidt
export orth_complement!
export orth_projection
include("BasisConstruction.jl")

export galerkin_projection
include("GalerkinProjections.jl")

export Projection
export PODBasis
export TTSVDCores
export BlockProjection
export ReducedProjection
export projection
export get_basis
export num_fe_dofs
export num_reduced_dofs
export get_cores
export project
export inv_project
export enrich
include("Projections.jl")

export FESubspace
export SingleFieldRBSpace
export MultiFieldRBSpace
export EvalFESubspace
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
export residual_snapshots
export jacobian_snapshots
include("PGOperator.jl")

export AbstractIntegrationDomain
export IntegrationDomain
export HyperReduction
export MDEIM
export AffineContribution
export empirical_interpolation
export integration_domain
export get_integration_domain
export reduce_triangulation
export reduced_jacobian
export reduced_residual
export reduced_weak_form
export inv_project!
include("HyperReduction.jl")

export PGMDEIMOperator
export LinearNonlinearPGMDEIMOperator
export fe_jacobian!
export fe_residual!
include("PGMDEIMOperator.jl")

export RBResults
export rb_results
export create_dir
export load_solve
export load_snapshots
export load_operator
export load_results
include("PostProcess.jl")

end # module
