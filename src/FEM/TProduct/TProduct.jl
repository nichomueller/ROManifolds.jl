module TProduct

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
using Gridap.MultiField
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Helpers

using Mabla.FEM.Utils
using Mabla.FEM.IndexMaps

import Base:+,-
import FillArrays: Fill,fill
import Kronecker: kronecker
import Gridap.ReferenceFEs: get_order
import PartitionedArrays: tuple_of_arrays
import SparseArrays: AbstractSparseMatrixCSC
import UnPack: @unpack

export get_dof_index_map
export get_polynomial_order
include("DofIndexMaps.jl")

export get_tp_dof_index_map
include("TProductDofIndexMaps.jl")

export TProductModel
export TProductTriangulation
export TProductMeasure
include("TProductGeometry.jl")

export TProductFESpace
export get_tp_fe_basis
export get_tp_trial_fe_basis
include("TProductFESpaces.jl")

export AbstractTProductTensor
export AbstractRank1Tensor
export AbstractRankTensor
export BlockGenericRankTensor
export get_factors
export get_decomposition
include("TProductTensor.jl")

export TProductCellPoint
export TProductCellField
export GenericTProductCellField
export TProductFEBasis
export GenericTProductDiffCellField
export GenericTProductDiffEval
export TProductCellDatum
include("TProductCellData.jl")

export TProductSparseMatrixAssembler
export TProductBlockSparseMatrixAssembler
include("TProductAssembly.jl")

end # module
