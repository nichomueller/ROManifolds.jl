module RBSteady

using LinearAlgebra
using LowRankApprox
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

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

using ReducedOrderModels.Utils
using ReducedOrderModels.IndexMaps
using ReducedOrderModels.TProduct
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamAlgebra
using ReducedOrderModels.ParamFESpaces
using ReducedOrderModels.ParamSteady
using ReducedOrderModels.ParamODEs

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
export Reduction
export DirectReduction
export GreedyReduction
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
export get_realization
export flatten_snapshots
export select_snapshots
export get_indexed_values
export num_space_dofs
include("Snapshots.jl")

export galerkin_projection
include("GalerkinProjections.jl")

export AbstractTTCore
export SparseCore
export SparseCoreCSC
include("TTCores.jl")

export TTContraction
export contraction
export sequential_product
export cores2basis
include("TTLinearAlgebra.jl")

export RBSolver
export RBOnlineCache
export get_fe_solver
export solution_snapshots
export residual_snapshots
export jacobian_snapshots
export nonlinear_rb_solve!
include("RBSolver.jl")

export reduction
export tpod
export ttsvd
export gram_schmidt
export orth_complement!
export orth_projection
include("BasisConstruction.jl")

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
export enrich!
include("Projections.jl")

export RBSpace
export SingleFieldRBSpace
export MultiFieldRBSpace
export EvalRBSpace
export EvalMultiFieldRBSpace
export fe_subspace
export reduced_fe_space
export reduced_basis
include("RBSpace.jl")

export AbstractIntegrationDomain
export IntegrationDomain
export HyperReduction
export EmptyHyperReduction
export MDEIM
export AffineContribution
export BlockHyperReduction
export HypRedCache
export empirical_interpolation
export integration_domain
export get_integration_domain
export reduced_triangulation
export reduced_jacobian
export reduced_residual
export reduced_weak_form
export inv_project!
include("HyperReduction.jl")

export RBOperator
export GenericRBOperator
export LinearNonlinearRBOperator
export reduced_operator
export get_fe_trial
export get_fe_test
export fe_jacobian!
export fe_residual!
include("ReducedOperators.jl")

export RBNewtonOperator
export RBNewtonOp
include("RBNewtonOperator.jl")

export RBPerformance
export rb_performance
export create_dir
export load_solve
export load_snapshots
export load_operator
export load_results
include("PostProcess.jl")

end # module
