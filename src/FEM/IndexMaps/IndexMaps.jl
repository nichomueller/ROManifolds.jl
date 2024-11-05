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
export get_nonzero_indices
include("IndexOperations.jl")

export SparsityPattern
export SparsityPatternCSC
export MultiValueSparsityPatternCSC
export TProductSparsityPattern
export get_sparsity
export permute_sparsity
include("SparsityPatterns.jl")

export AbstractIndexMap
export AbstractTrivialIndexMap
export TrivialIndexMap
export TrivialSparseIndexMap
export IndexMap
export IndexMapView
export ShowSlaveDofsStyle
export ShowSlaveDofs
export DoNotShowSlaveDofs
export ConstrainedDofsIndexMap
export AbstractMultiValueIndexMap
export MultiValueIndexMap
export MultiValueIndexMapView
export TProductIndexMap
export SparseIndexMap
export MultiValueSparseIndexMap
export get_index_map
export inv_index_map
export vectorize_map
export remove_constrained_dofs
export change_index_map
export get_sparse_index_map
export recast
export get_fixed_dofs
export compose_indices
export get_component
export merge_components
export split_components
include("IndexMapsInterface.jl")

export FEOperatorIndexMap
export get_vector_index_map
export get_matrix_index_map
include("FEIndexMaps.jl")

end # module
