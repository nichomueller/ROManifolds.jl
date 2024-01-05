module RB
using Mabla.Utils
using Mabla.FEM
using LinearAlgebra
using SparseArrays
using Serialization
using NearestNeighbors
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools

import StaticArrays: SVector
import UnPack: @unpack
import Gridap.Helpers: @check
import Gridap.Helpers: @unreachable
import Gridap.Arrays: evaluate!
import Gridap.Algebra: allocate_matrix
import Gridap.Algebra: allocate_vector
import Gridap.Algebra: solve

export RBInfo
export BlockRBInfo
export ComputationInfo
export get_norm_matrix
export Snapshots
export num_space_dofs
export num_time_dofs
export num_params
export collect_solutions
export NnzArray
export NnzVector
export NnzMatrix
export get_nonzero_val
export get_nonzero_idx
export get_nrows
export recast
export compress
export collect_residuals_for_trian
export collect_jacobians_for_trian
export RBSpace
export reduced_basis
export get_basis_space
export get_basis_time
export num_rb_space_ndofs
export num_rb_time_ndofs
export num_rb_ndofs
export space_time_projection
export RBAffineDecomposition
export RBVecAffineDecomposition
export RBMatAffineDecomposition
export GenericRBAffineDecomposition
export TrivialRBAffineDecomposition
export RBIntegrationDomain
export RBContributionMap
export RBVecContributionMap
export RBMatContributionMap
export get_interpolation_idx
export project_space
export project_time
export project_space_time
export get_reduced_cells
export collect_reduced_residuals!
export collect_reduced_jacobians!
export rb_coefficient!
export rb_contribution!
export zero_rb_contribution
export RBAlgebraicContribution
export RBVecAlgebraicContribution
export RBMatAlgebraicContribution
export collect_compress_rhs
export collect_compress_lhs
export collect_compress_rhs_lhs
export compress_component
export collect_rhs_contributions!
export collect_lhs_contributions!
export collect_rhs_lhs_contributions!
export RBResults
export rb_solver
export post_process
export plot_results
export compute_relative_error
export nearest_neighbor
export RBBlock
export BlockSnapshots
export BlockRBSpace
export BlockRBAlgebraicContribution
export BlockRBVecAlgebraicContribution
export BlockRBMatAlgebraicContribution
export BlockRBResults
export get_blocks
export fe_offsets
export rb_offsets
export add_space_supremizers
export add_time_supremizers

include("RBInfo.jl")
include("Snapshots.jl")
include("NnzArrays.jl")
include("RBSpaces.jl")
include("RBAffineDecomposition.jl")
include("RBAlgebraicContribution.jl")
include("RBResults.jl")
include("RBBlocks.jl")
end # module
