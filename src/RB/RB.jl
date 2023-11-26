module
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

import StaticArrays: SVector
import UnPack: @unpack
import Gridap.Helpers: @check,@unreachable
import Gridap.Arrays: evaluate!
import Gridap.Algebra: allocate_matrix,allocate_vector,solve
import Gridap.ODEs.ODETools: solve_step!
import Gridap.ODEs.TransientFETools: ODESolver

export RBInfo,BlockRBInfo,ComputationInfo,get_norm_matrix
export Snapshots,num_space_dofs,num_time_dofs,num_params
export NnzArray,NnzVector,NnzMatrix,get_nonzero_val,get_nonzero_idx,get_nrows,recast,compress
export RBSpace,reduced_basis,get_basis_space,get_basis_time,num_rb_space_ndofs,num_rb_time_ndofs,num_rb_ndofs,space_time_projection
export RBAffineDecomposition,RBVecAffineDecomposition,RBMatAffineDecomposition,GenericRBAffineDecomposition,TrivialRBAffineDecomposition,RBIntegrationDomain,RBContributionMap,RBVecContributionMap,RBMatContributionMap,get_interpolation_idx,project_space,project_time,project_space_time,get_reduced_cells,collect_reduced_residuals!,collect_reduced_jacobians!,rb_coefficient!,rb_contribution!,zero_rb_contribution
export RBAlgebraicContribution,RBVecAlgebraicContribution,RBMatAlgebraicContribution,collect_compress_rhs,collect_compress_lhs,collect_compress_rhs_lhs,compress_component,collect_rhs_contributions!,collect_lhs_contributions!,collect_rhs_lhs_contributions!
export RBResults,rb_solver,post_process,plot_results,compute_relative_error,nearest_neighbor
export RBBlock,BlockSnapshots,BlockRBSpace,BlockRBAlgebraicContribution,BlockRBVecAlgebraicContribution,BlockRBMatAlgebraicContribution,BlockRBResults,get_blocks,get_nblocks,fe_offsets,rb_offsets,add_space_supremizers,add_time_supremizers

include("RBInfo.jl")
include("Snapshots.jl")
include("NnzArrays.jl")
include("RBSpaces.jl")
include("RBAffineDecomposition.jl")
include("RBAlgebraicContribution.jl")
include("RBResults.jl")
include("RBBlocks.jl")
end # module
