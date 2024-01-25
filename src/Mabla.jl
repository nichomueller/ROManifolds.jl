module Mabla

include("Utils/Utils.jl")
using Mabla.Utils

export Float
export get_parent_dir
export create_dir
export correct_path
export save
export load
export num_active_dirs
export vec_to_mat_idx
export slow_idx
export fast_idx
export index_pairs
export change_mode
export compress_array
export recenter
export tpod
export gram_schmidt!
export orth_complement!
export orth_projection

include("FEM/FEM.jl")
using Mabla.FEM

export Table
export Affine
export ParametricSpace
export UniformSampling
export NormalSampling
export realization
export ParamArray
export allocate_param_array
export zero_param_array
export get_at_offsets
export recenter
export TrialPFESpace
export TrialPFESpace!
export HomogeneousTrialPFESpace
export MultiFieldPFESpace
export AbstractPFunction
export PFunction, 𝑓ₚ
export TransientPFunction, 𝑓ₚₜ
export TransientTrialPFESpace
export TransientMultiFieldPFESpace
export TransientMultiFieldTrialPFESpace
export TransientPFEOperator
export TransientPFEOperatorFromWeakForm
export AffineTransientPFEOperator
export NonlinearTransientPFEOperator
export residual_for_trian!
export jacobian_for_trian!
export ODEPSolution
# export num_time_dofs
# export get_times
# export get_stencil_times
export collect_solutions
export collect_residuals_for_trian
export collect_jacobians_for_trian
export AffineThetaMethodPOperator
export ThetaMethodPOperator
export get_order
export get_L2_norm_matrix
export get_H1_norm_matrix
# export ReducedMeasure
export PString
export PVisualizationData

# include("RB/RB.jl")
# using Mabla.RB

# export RBInfo
# export BlockRBInfo
# export ComputationInfo
# export get_norm_matrix
# export Snapshots
# export num_space_dofs
# export num_time_dofs
# export num_params
# export NnzArray
# export NnzVector
# export NnzMatrix
# export get_nonzero_val
# export get_nonzero_idx
# export get_nrows
# export recast
# export compress
# export RBSpace
# export reduced_basis
# export get_basis_space
# export get_basis_time
# export num_rb_space_ndofs
# export num_rb_time_ndofs
# export num_rb_ndofs
# export space_time_projection
# export RBAffineDecomposition
# export RBVecAffineDecomposition
# export RBMatAffineDecomposition
# export GenericRBAffineDecomposition
# export TrivialRBAffineDecomposition
# export RBIntegrationDomain
# export RBContributionMap
# export RBVecContributionMap
# export RBMatContributionMap
# export get_interpolation_idx
# export project_space
# export project_time
# export project_space_time
# export get_reduced_cells
# export collect_reduced_residuals!
# export collect_reduced_jacobians!
# export rb_coefficient!
# export rb_contribution!
# export zero_rb_contribution
# export RBAlgebraicContribution
# export RBVecAlgebraicContribution
# export RBMatAlgebraicContribution
# export collect_compress_rhs
# export collect_compress_lhs
# export collect_compress_rhs_lhs
# export compress_component
# export collect_rhs_contributions!
# export collect_lhs_contributions!
# export collect_rhs_lhs_contributions!
# export RBResults
# export rb_solver
# export post_process
# export plot_results
# export compute_relative_error
# export nearest_neighbor
# export RBBlock
# export BlockSnapshots
# export BlockRBSpace
# export BlockRBAlgebraicContribution
# export BlockRBVecAlgebraicContribution
# export BlockRBMatAlgebraicContribution
# export BlockRBResults
# export get_blocks
# export fe_offsets
# export rb_offsets
# export add_space_supremizers
# export add_time_supremizers

# include("Distributed/Distributed.jl")
# using Mabla.Distributed
end # module
