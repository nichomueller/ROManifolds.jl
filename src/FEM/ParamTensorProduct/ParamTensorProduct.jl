module ParamTensorProduct

using LinearAlgebra
using BlockArrays
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Helpers

using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamODEs

import FillArrays: Fill,fill
import Kronecker: kronecker
import Gridap.ReferenceFEs: get_order
import Gridap.TensorValues: Mutable
import PartitionedArrays: tuple_of_arrays
import SparseArrays: AbstractSparseMatrixCSC

export SparsityPattern
export SparsityPatternCSC
export TProductSparsityPattern
export get_sparsity
include("SparsityPatterns.jl")

export AbstractIndexMap
export IndexMap
export IndexMapView
export MultiValueIndexMap
export FixedDofIndexMap
export TProductIndexMap
export SparseIndexMap
export inv_index_map
export free_dofs_map
export vectorize_index_map
export recast_indices
export sparsify_indices
export get_nonzero_indices
export tensorize_indices
export split_row_col_indices
include("TProductIndexMaps.jl")

export TProductModel
export TProductTriangulation
export TProductMeasure
include("TProductGeometry.jl")

export TProductFESpace
export TProductFEBasis
export get_dof_permutation
export get_sparse_index_map
export get_tp_dof_permutation
export get_tp_fe_basis
export get_tp_trial_fe_basis
include("TProductFESpaces.jl")

export TProductCellPoint
export TProductCellFields
export TProductGradientCellField
export TProductGradientEval
export TProductSparseMatrixAssembler
export AbstractTProductArray
export TProductArray
export TProductGradientArray
export symbolic_kron
export symbolic_kron!
export numerical_kron!
export kronecker_gradients
include("TProductCellFields.jl")

export TTArray
export TTVector
export TTMatrix
export ParamTTArray
export ParamTTVector
export ParamTTMatrix
export ParamTTSparseMatrix
export get_values
export get_index_map
include("TTArray.jl")

export ParamBlockTTArray
export ParamBlockTTVector
export ParamBlockTTMatrix
include("ParamBlockTTArray.jl")

export TTBuilder
export TTCounter
export TTInserter
export ParamTTInserterCSC
include("TTAlgebra.jl")

end # module
