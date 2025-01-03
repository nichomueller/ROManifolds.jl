module ROM

export PerformanceTracker
export CostTracker
export SU
export reset_tracker!
export update_tracker!
export compute_speedup
export compute_error
export compute_relative_error
export induced_norm

export PartialFunctions
export PartialDerivative
export PartialTrace
export ∂₁, ∂₂, ∂₃

export get_values
export get_parent
export order_domains

export Contribution
export ArrayContribution
export VectorContribution
export MatrixContribution
export TupOfArrayContribution
export contribution
export set_domains
export change_domains

include("FEM/Utils/Utils.jl")
using ROM.Utils

export recast_indices
export get_nonzero_indices
export slow_index
export fast_index

export SparsityPattern
export SparsityCSC
export MultiValueSparsityPatternCSC
export TProductSparsity
export order_sparsity

export AbstractDofMap
export AbstractTrivialDofMap
export TrivialDofMap
export TrivialSparseDofMap
export DofMap
export DofMapView
export AbstractMultiValueDofMap
export MultiValueDofMap
export MultiValueDofMapView
export TProductDofMap
export SparseDofMap
export MultiValueSparseDofMap
export invert
export remove_constrained_dofs
export get_sparse_dof_map
export recast
export get_component

export FEDofMap
export get_dof_map
export get_sparse_dof_map

include("FEM/DofMaps/DofMaps.jl")
using ROM.DofMaps

export get_tp_dof_map

export TProductModel
export TProductTriangulation
export TProductMeasure

export TProductFESpace
export get_tp_fe_basis
export get_tp_trial_fe_basis

export AbstractRankTensor
export Rank1Tensor
export GenericRankTensor
export BlockRankTensor
export get_factors
export get_decomposition

export TProductCellPoint
export TProductCellField
export GenericTProductCellField
export TProductFEBasis
export GenericTProductDiffCellField
export GenericTProductDiffEval
export TProductCellDatum

export TProductSparseMatrixAssembler
export TProductBlockSparseMatrixAssembler

include("FEM/TProduct/TProduct.jl")
using ROM.TProduct

export AbstractRealization
export Realization
export TransientRealization
export UniformSampling
export NormalSampling
export HaltonSampling
export ParamSpace
export TransientParamSpace
export AbstractParamFunction
export ParamFunction, 𝑓ₚ
export TransientParamFunction, 𝑓ₚₜ
export realization
export get_params
export get_times
export get_initial_time
export get_final_time
export num_params
export num_times
export shift!

export AbstractParamContainer
export ParamType
export ParamContainer
export ParamNumber
export get_param_data
export param_length
export param_eachindex
export param_getindex
export param_setindex!

export ParamField
export ParamFieldGradient
export GenericParamField
export OperationParamField

export AbstractParamArray
export AbstractParamVector
export AbstractParamMatrix
export ParamArray
export ParamVector
export ParamMatrix
export param_array

export TrivialParamArray
export ConsecutiveParamArray
export ConsecutiveParamVector
export ConsecutiveParamMatrix
export GenericParamVector
export GenericParamMatrix

export ConsecutiveParamVector
export ConsecutiveParamMatrix

export ParamSparseMatrix
export ParamSparseMatrixCSC
export ParamSparseMatrixCSR
export ConsecutiveParamSparseMatrixCSC
export GenericParamSparseMatrixCSC
export ConsecutiveParamSparseMatrixCSR
export GenericParamSparseMatrixCSR

export BlockParamArray
export BlockParamVector
export BlockParamMatrix
export BlockConsecutiveParamVector
export BlockConsecutiveParamMatrix

include("FEM/ParamDataStructures/ParamDataStructures.jl")
using ROM.ParamDataStructures

export ParamCounter
export ParamInserterCSC
export eltype2

include("FEM/ParamGeometry/ParamGeometry.jl")
using ROM.ParamGeometry

include("FEM/ParamAlgebra/ParamAlgebra.jl")
using ROM.ParamAlgebra

export SingleFieldParamFESpace
export TrivialParamFESpace

export MultiFieldParamFESpace

export TrialParamFESpace
export TrialParamFESpace!
export HomogeneousTrialParamFESpace

export ParamFEFunction
export SingleFieldParamFEFunction
export MultiFieldParamFEFunction

export get_param_assembler
export collect_cell_matrix_for_trian
export collect_cell_vector_for_trian

include("FEM/ParamFESpaces/ParamFESpaces.jl")
using ROM.ParamFESpaces

export UnEvalTrialFESpace
export ParamTrialFESpace
export ParamMultiFieldFESpace

export UnEvalOperatorType
export NonlinearParamEq
export LinearParamEq
export LinearNonlinearParamEq
export ParamOperator
export LinearNonlinearParamOpFromFEOp

export ParamFEOperator
export LinearParamFEOperator
export FEDomains
export set_domains
export change_domains

export LinearNonlinearParamODE
export LinearNonlinearParamFEOperator
export get_linear_operator
export get_nonlinear_operator
export join_operators

export ParamOpFromFEOp
export JointParamOpFromFEOp
export SplitParamOpFromFEOp

include("FEM/ParamSteady/ParamSteady.jl")
using ROM.ParamSteady

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace

export ODEParamOperatorType
export NonlinearParamODE
export LinearParamODE
export ODEParamOperator

export ParamStageOperator

export TransientParamFEOperator
export TransientParamFEOpFromWeakForm
export TransientParamLinearFEOperator
export TransientParamLinearFEOpFromWeakForm
export LinearTransientParamFEOperator
export NonlinearTransientParamFEOperator

export LinearNonlinearParamODE
export LinearNonlinearTransientParamFEOperator
export get_linear_operator
export get_nonlinear_operator
export join_operators

export ODEParamOpFromTFEOp

export ODEParamSolution

export TransientParamFESolution

include("FEM/ParamODEs/ParamODEs.jl")
using ROM.ParamODEs

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

export AbstractSnapshots
export Snapshots
export GenericSnapshots
export SnapshotsAtIndices
export SparseSnapshots
export UnfoldingSteadySnapshots
export BlockSnapshots
export get_realization
export flatten_snapshots
export select_snapshots
export get_indexed_data
export num_space_dofs

export galerkin_projection

export AbstractTTCore
export SparseCore
export SparseCoreCSC

export TTContraction
export contraction
export sequential_product
export cores2basis

export RBSolver
export RBOnlineCache
export get_fe_solver
export solution_snapshots
export residual_snapshots
export jacobian_snapshots
export nonlinear_rb_solve!

export reduction
export tpod
export ttsvd
export gram_schmidt
export orth_complement!
export orth_projection

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

export RBSpace
export SingleFieldRBSpace
export MultiFieldRBSpace
export EvalRBSpace
export EvalMultiFieldRBSpace
export fe_subspace
export reduced_fe_space
export reduced_basis

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

export RBOperator
export GenericRBOperator
export LinearNonlinearRBOperator
export reduced_operator
export get_fe_trial
export get_fe_test
export fe_jacobian!
export fe_residual!

export ROMPerformance
export eval_performance
export create_dir
export load_solve
export load_snapshots
export load_operator
export load_results

include("RB/RBSteady/RBSteady.jl")
using ROM.RBSteady

export TransientReduction
export TransientMDEIMReduction

export TransientSnapshots
export TransientGenericSnapshots
export TransientSparseSnapshots
export UnfoldingTransientSnapshots
export ModeTransientSnapshots

export TransientProjection

export TransientRBSpace
export TransientEvalMultiFieldRBSpace

export TransientMDEIM
export TransientAffineDecomposition

export TransientRBOperator
export GenericTransientRBOperator
export LinearNonlinearTransientRBOperator

include("RB/RBTransient/RBTransient.jl")
using ROM.RBTransient

# include("Distributed/Distributed.jl")
# using ROM.Distributed
end
