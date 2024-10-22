module ReducedOrderModels

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
export ∂ₓ₁, ∂ₓ₂, ∂ₓ₃

include("FEM/Utils/Utils.jl")
using ReducedOrderModels.Utils

export recast_indices
export get_nonzero_indices

export SparsityPattern
export SparsityPatternCSC
export MultiValueSparsityPatternCSC
export TProductSparsityPattern
export get_sparsity
export permute_sparsity

export AbstractIndexMap
export AbstractTrivialIndexMap
export TrivialIndexMap
export TrivialSparseIndexMap
export IndexMap
export IndexMapView
export FixedDofsIndexMap
export AbstractMultiValueIndexMap
export MultiValueIndexMap
export MultiValueIndexMapView
export TProductIndexMap
export SparseIndexMap
export MultiValueSparseIndexMap
export get_index_map
export inv_index_map
export free_dofs_map
export change_index_map
export get_sparse_index_map
export recast
export get_fixed_dofs
export remove_fixed_dof
export compose_indices
export get_component
export merge_components
export split_components

export FEOperatorIndexMap
export get_vector_index_map
export get_matrix_index_map

include("FEM/IndexMaps/IndexMaps.jl")
using ReducedOrderModels.IndexMaps

export get_dof_index_map
export get_polynomial_order

export get_tp_dof_index_map

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
using ReducedOrderModels.TProduct

export AbstractRealization
export Realization
export TransientRealization
export UniformSampling
export NormalSampling
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
export slow_index
export fast_index
export shift!

export AbstractParamContainer
export ParamContainer
export ParamNumber
export get_param_data
export param_length
export param_eachindex
export param_getindex
export param_setindex!
export get_param_entry

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

export Contribution
export ArrayContribution
export VectorContribution
export MatrixContribution
export TupOfArrayContribution
export contribution
export get_values
export get_parent
export order_triangulations
export find_closest_view

include("FEM/ParamDataStructures/ParamDataStructures.jl")
using ReducedOrderModels.ParamDataStructures

export ParamCounter
export ParamInserterCSC
export eltype2

include("FEM/ParamAlgebra/ParamAlgebra.jl")
using ReducedOrderModels.ParamAlgebra

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
using ReducedOrderModels.ParamFESpaces

export UnEvalParamSingleFieldFESpace
export ParamTrialFESpace
export ParamMultiFieldFESpace

export UnEvalOperatorType
export NonlinearParamEq
export LinearParamEq
export LinearNonlinearParamEq
export ParamOperator
export ParamOperatorWithTrian
export LinearNonlinearParamOpFromFEOp
export ParamCache
export allocate_paramcache
export update_paramcache!
export get_realization

export ParamFEOperator
export LinearParamFEOperator
export ParamFEOperatorWithTrian
export ParamFEOpFromWeakFormWithTrian
export set_triangulation
export change_triangulation

export LinearNonlinearParamODE
export LinNonlinParamFEOperator
export LinearNonlinearParamFEOperator
export LinearNonlinearParamFEOperatorWithTrian
export get_linear_operator
export get_nonlinear_operator
export join_operators

export ParamOpFromFEOp
export ParamOpFromFEOpWithTrian

include("FEM/ParamSteady/ParamSteady.jl")
using ReducedOrderModels.ParamSteady

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace

export ODEParamOperatorType
export NonlinearParamODE
export AbstractLinearParamODE
export SemilinearParamODE
export LinearParamODE
export ODEParamOperator
export ODEParamOperatorWithTrian

export ParamStageOperator
export NonlinearParamStageOperator
export LinearParamStageOperator

export TransientParamFEOperator
export TransientParamFEOpFromWeakForm
export TransientParamSemilinearFEOperator
export TransientParamSemilinearFEOpFromWeakForm
export TransientParamLinearFEOperator
export TransientParamLinearFEOpFromWeakForm
export LinearTransientParamFEOperator
export NonlinearTransientParamFEOperator

export TransientParamFEOperatorWithTrian
export TransientParamFEOpFromWeakFormWithTrian
export set_triangulation
export change_triangulation

export LinearNonlinearParamODE
export LinNonlinTransientParamFEOperator
export LinearNonlinearTransientParamFEOperator
export LinearNonlinearTransientParamFEOperatorWithTrian
export get_linear_operator
export get_nonlinear_operator
export join_operators

export ODEParamOpFromTFEOp
export ODEParamOpFromTFEOpWithTrian

export ODEParamSolution

export TransientParamFESolution

include("FEM/ParamODEs/ParamODEs.jl")
using ReducedOrderModels.ParamODEs

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
export GenericSnapshots
export SnapshotsAtIndices
export SparseSnapshots
export UnfoldingSteadySnapshots
export BlockSnapshots
export Snapshots
export flatten_snapshots
export select_snapshots
export get_indexed_values
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

export FESubspace
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

export RBNewtonOperator
export RBNewtonOp

export RBPerformance
export rb_performance
export create_dir
export load_solve
export load_snapshots
export load_operator
export load_results

include("RB/RBSteady/RBSteady.jl")
using ReducedOrderModels.RBSteady

export TransientReduction
export TransientMDEIMReduction

export AbstractTransientSnapshots
export TransientGenericSnapshots
export TransientSparseSnapshots
export UnfoldingTransientSnapshots
export ModeTransientSnapshots

export TransientBasis

export TransientRBSpace
export TransientEvalMultiFieldRBSpace

export TransientMDEIM
export TransientAffineDecomposition

export TransientRBOperator
export GenericTransientRBOperator
export LinearNonlinearTransientRBOperator

export TransientRBNewtonOp

include("RB/RBTransient/RBTransient.jl")
using ReducedOrderModels.RBTransient

# include("Distributed/Distributed.jl")
# using ReducedOrderModels.Distributed
end
