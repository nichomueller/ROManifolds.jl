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

using GridapEmbedded
using GridapEmbedded.Interfaces

using ROM.Utils
using ROM.DofMaps

import Base:+,-
import FillArrays: Fill,fill
import Gridap.ReferenceFEs: get_order
import PartitionedArrays: tuple_of_arrays
import SparseArrays: AbstractSparseMatrixCSC
import UnPack: @unpack

export get_tp_dof_map
include("TProductDofMaps.jl")

export TProductModel
export TProductTriangulation
export TProductMeasure
include("TProductGeometry.jl")

export TProductFESpace
export get_tp_fe_basis
export get_tp_trial_fe_basis
include("TProductFESpaces.jl")

export AbstractRankTensor
export Rank1Tensor
export GenericRankTensor
export BlockRankTensor
export MatrixOrTensor
export get_factors
export get_decomposition
include("RankTensors.jl")

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

export OrderedFESpace
export OIdsToIds
export DofsToODofs
export add_ordered_entries!
include("OrderedFESpaces.jl")

end # module
