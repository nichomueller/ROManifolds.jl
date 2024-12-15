module DofMaps

using LinearAlgebra
using SparseArrays

using Gridap
using Gridap.Arrays
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Helpers

using GridapEmbedded
using GridapEmbedded.Interfaces

using ReducedOrderModels
using ReducedOrderModels.Utils

import FillArrays: Fill
import PartitionedArrays: tuple_of_arrays
import SparseArrays: AbstractSparseMatrix,AbstractSparseMatrixCSC
import SparseMatricesCSR: SparseMatrixCSR

export recast_indices
export get_nonzero_indices
export slow_index
export fast_index
include("IndexOperations.jl")

export Range2D
export range_1d
export range_2d
include("Ranges.jl")

export AbstractDofMap
export AbstractTrivialDofMap
export TrivialDofMap
export TrivialSparseDofMap
export DofMap
export ConstrainedDofMap
export DofMapPortion
export TProductDofMap
export invert
export vectorize
include("DofMapsInterface.jl")

export SparsityPattern
export SparsityCSC
export OrderedSparsity
export TProductSparsityPattern
export TProductSparsity
export SparsityToTProductSparsity
export get_background_sparsity
export order_sparsity
export recast
export get_matrix_to_parent_matrix
include("SparsityPatterns.jl")

export SparseDofMapStyle
export FullDofMapIndexing
export SparseDofMapIndexing
export TrivialSparseDofMap
export SparseDofMap
export get_sparsity
include("SparseDofMaps.jl")

export get_dof_map
export get_dof_type
export get_sparse_dof_map
export get_univariate_dof_map
export get_dirichlet_entities
export get_polynomial_order
export get_tface_to_mask
include("DofMapsBuilders.jl")

export FEDofMap
include("FEDofMaps.jl")

export DofMapArray
include("DofMapArrays.jl")

end # module
