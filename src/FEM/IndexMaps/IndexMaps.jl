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
import PartitionedArrays: tuple_of_arrays

export recast_indices
export sparsify_indices
export get_nonzero_indices
export tensorize_indices
include("IndexOperations.jl")

export SparsityPattern
export SparsityPatternCSC
export MultiValueSparsityPatternCSC
export TProductSparsityPattern
export get_sparsity
export permute_sparsity
include("SparsityPatterns.jl")

export FixedEntriesArray
include("FixedEntriesArrays.jl")

export AbstractIndexMap
export TrivialIndexMap
export IndexMap
export IndexMapView
export FixedDofsIndexMap
export TProductIndexMap
export SparseIndexMap
export AbstractMultiValueIndexMap
export MultiValueIndexMap
export get_index_map
export inv_index_map
export free_dofs_map
export change_index_map
export get_sparse_index_map
export recast
export remove_fixed_dof
export compose_indices
export get_component
export split_components
export merge_components
include("IndexMapsInterface.jl")

end # module
