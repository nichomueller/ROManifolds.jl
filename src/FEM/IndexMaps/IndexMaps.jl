module DofMaps

using LinearAlgebra
using SparseArrays

using Gridap
using Gridap.Arrays
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.TensorValues
using Gridap.Helpers

using ReducedOrderModels
using ReducedOrderModels.Utils

import SparseArrays: AbstractSparseMatrix
import PartitionedArrays: tuple_of_arrays

export recast_indices
export get_nonzero_indices
export slow_index
export fast_index
include("IndexOperations.jl")

export SparsityPattern
export SparsityPatternCSC
export MultiValueSparsityPatternCSC
export TProductSparsityPattern
export order_sparsity
export to_nz_index
export to_nz_index!
include("SparsityPatterns.jl")

export AbstractDofMap
export AbstractTrivialDofMap
export TrivialDofMap
export TrivialSparseDofMap
export DofMap
export DofMapView
export ShowSlaveDofsStyle
export ShowSlaveDofs
export DoNotShowSlaveDofs
export ConstrainedDofsDofMap
export AbstractMultiValueDofMap
export MultiValueDofMap
export MultiValueDofMapView
export TProductDofMap
export SparseDofMap
export MultiValueSparseDofMap
export invert
export vectorize_map
export remove_constrained_dofs
export get_sparse_dof_map
export recast
export get_fixed_dofs
export get_component
include("DofMapsInterface.jl")

export FEOperatorDofMap
export get_vector_dof_map
export get_matrix_dof_map
include("FEDofMaps.jl")

end # module
