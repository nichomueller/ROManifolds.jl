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
import SparseArrays: AbstractSparseMatrix
import SparseMatricesCSR: SparseMatrixCSR

export recast_indices
export get_nonzero_indices
export slow_index
export fast_index
include("IndexOperations.jl")

export AbstractDofMap
export AbstractTrivialDofMap
export TrivialDofMap
export TrivialSparseDofMap
export DofMap
export ConstrainedDofMap
export TProductDofMap
export invert
export vectorize
include("DofMapsInterface.jl")

export SparsityPattern
export SparsityPatternCSC
export TProductSparsityPattern
export order_sparsity
export recast
export to_nz_index
export to_nz_index!
include("SparsityPatterns.jl")

export SparseDofMapStyle
export FullDofMapIndexing
export SparseDofMapIndexing
export TrivialSparseDofMap
export SparseDofMap
include("SparseDofMaps.jl")

export get_dof_map
export get_sparse_dof_map
export get_polynomial_order
include("DofMapsBuilders.jl")

export FEDofMap
include("FEDofMaps.jl")

end # module
