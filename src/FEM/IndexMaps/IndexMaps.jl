module IndexMaps

using LinearAlgebra
using SparseArrays

using Gridap
using Gridap.Arrays
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.TensorValues
using Gridap.Helpers

import SparseArrays: AbstractSparseMatrix

export recast_indices
export sparsify_indices
export get_nonzero_indices
export tensorize_indices
export split_row_col_indices
include("IndexOperations.jl")

export SparsityPattern
export SparsityPatternCSC
export MultiValueSparsityPatternCSC
export TProductSparsityPattern
export get_sparsity
export permute_sparsity
include("SparsityPatterns.jl")

export AbstractIndexMap
export TrivialIndexMap
export IndexMap
export IndexMapView
export MultiValueIndexMap
export FixedDofIndexMap
export TProductIndexMap
export SparseIndexMap
export get_index_map
export inv_index_map
export free_dofs_map
export change_index_map
include("IndexMapsInterface.jl")

end # module
