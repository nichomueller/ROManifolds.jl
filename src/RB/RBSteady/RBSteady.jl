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

using ROM.Utils
using ROM.DofMaps
using ROM.TProduct
using ROM.ParamDataStructures
using ROM.ParamAlgebra
using ROM.ParamFESpaces
using ROM.ParamSteady
using ROM.ParamODEs

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
export Snapshots
export GenericSnapshots
export SnapshotsAtIndices
export SparseSnapshots
export BlockSnapshots
export get_realization
export flatten_snapshots
export select_snapshots
export get_indexed_data
export num_space_dofs
include("Snapshots.jl")

export galerkin_projection
include("GalerkinProjections.jl")

export AbstractTTCore
export DofMapCore
export SparseCore
export SparseCoreCSC
include("TTCores.jl")

export TTContraction
export contraction
export unbalanced_contractions
export sequential_product
export cores2basis
include("TTLinearAlgebra.jl")

export RBSolver
export get_fe_solver
export solution_snapshots
export residual_snapshots
export jacobian_snapshots
include("RBSolver.jl")

export reduction
export tpod
export ttsvd
export gram_schmidt
export orth_complement!
export orth_projection
include("BasisConstruction.jl")

export Projection
export PODProjection
export TTSVDProjection
export NormedProjection
export BlockProjection
export ReducedProjection
export projection
export get_basis
export num_fe_dofs
export num_reduced_dofs
export get_cores
export project
export project!
export inv_project
export inv_project!
export union_bases
export enrich!
include("Projections.jl")

export RBSpace
export SingleFieldRBSpace
export MultiFieldRBSpace
export SingleFieldParamRBSpace
export reduced_subspace
export reduced_spaces
export reduced_basis
export get_reduced_subspace
include("RBSpace.jl")

export RBParamVector
include("RBParamVector.jl")

export AbstractIntegrationDomain
export IntegrationDomain
export HyperReduction
export EmptyHyperReduction
export MDEIM
export AffineContribution
export BlockHyperReduction
export empirical_interpolation
export integration_domain
export get_integration_domain
export reduced_triangulation
export reduced_jacobian
export reduced_residual
export reduced_weak_form
export inv_project!
export allocate_hypred_cache
include("HyperReduction.jl")

export HRParamArray
include("HRParamArray.jl")

export RBOperator
export GenericRBOperator
export LinearNonlinearRBOperator
export RBCache
export LinearNonlinearRBCache
export reduced_operator
export get_fe_trial
export get_fe_test
export fe_jacobian!
export fe_residual!
export allocate_rbcache
include("ReducedOperators.jl")

export ROMPerformance
export eval_performance
export eval_convergence
export create_dir
export load_snapshots
export load_residuals
export load_jacobians
export load_contribution
export load_operator
export load_results
export plot_a_solution
include("PostProcess.jl")

end # module
