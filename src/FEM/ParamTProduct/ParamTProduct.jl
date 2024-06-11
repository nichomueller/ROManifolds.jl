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

export get_dof_index_map
export get_polynomial_order
include("DofIndexMaps.jl")

export get_tp_dof_index_map
include("TProductDofIndexMaps.jl")

export get_sparse_index_map
include("FEOperatorIndexMaps.jl")

export TProductModel
export TProductTriangulation
export TProductMeasure
include("TProductGeometry.jl")

export TProductFESpace
export get_tp_fe_basis
export get_tp_trial_fe_basis
include("TProductFESpaces.jl")

export AbstractTProductArray
export TProductArray
export TProductGradientArray
export symbolic_kron
export symbolic_kron!
export numerical_kron!
export kronecker_gradients
include("TProductArray.jl")

export TProductCellPoint
export TProductCellFields
export TProductFEBasis
export TProductGradientCellField
export TProductGradientEval
include("TProductCellData.jl")

export TProductSparseMatrixAssembler
include("TProductAssembly.jl")
end # module
