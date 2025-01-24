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

using ROM
using ROM.Utils

import FillArrays: Fill
import PartitionedArrays: tuple_of_arrays
import SparseArrays: AbstractSparseMatrix,AbstractSparseMatrixCSC
import SparseMatricesCSR: SparseMatrixCSR

export recast_indices
export slow_index
export fast_index
include("IndexOperations.jl")

export Range2D
export range_1d
export range_2d
include("Ranges.jl")

export OIdsToIds
export DofsToODofs
export add_ordered_entries!
include("OrderingMaps.jl")

export SparsityPattern
export SparsityCSC
export TProductSparsity
export get_sparsity
export get_background_sparsity
export recast
export get_d_sparse_dofs_to_full_dofs
include("SparsityPatterns.jl")

export AbstractDofMap
export TrivialDofMap
export VectorDofMap
export TrivialSparseMatrixDofMap
export SparseMatrixDofMap
export invert
export vectorize
export flatten
include("DofMapsInterface.jl")

export get_dof_map
export get_sparse_dof_map
export get_polynomial_order
export get_polynomial_orders
include("DofMapsBuilders.jl")

export DofMapArray
include("DofMapArrays.jl")

export OrderedFESpace
export CartesianFESpace
export get_ordered_cell_dof_ids
export get_bg_dof_to_mask
include("OrderedFESpaces.jl")

end # module
