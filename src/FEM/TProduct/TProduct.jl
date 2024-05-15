module TProduct
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.Helpers
using Gridap.MultiField
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Mabla.FEM

import FillArrays: Fill,fill
import IterTools: subsets
import Kronecker: kronecker
import OneHotArrays: OneHotMatrix,OneHotVector
import Test: @test
import UnPack: @unpack
import Gridap.Fields: OperationField,BroadcastOpFieldArray,BroadcastingFieldOpMap,LinearCombinationField,LinearCombinationMap
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ReferenceFEs: get_order
import Gridap.TensorValues: Mutable,inner,outer,double_contraction,symmetric_part
import LinearAlgebra: det,tr,cross,dot,⋅,rmul!
import Base: inv,abs,abs2,*,+,-,/,adjoint,transpose,real,imag,conj
import SparseArrays: AbstractSparseMatrixCSC
import PartitionedArrays: tuple_of_arrays
import Mabla.FEM: get_dirichlet_cells

export TProductModel
export TProductTriangulation
export TProductMeasure
include("TProductGeometry.jl")

export TProductFESpace
export TProductFEBasis
export get_dof_permutation
export get_tp_dof_permutation
export get_tp_fe_basis
export get_tp_trial_fe_basis
include("TProductFESpaces.jl")

export TProductCellPoint
export TProductCellFields
export TProductGradientCellField
export TProductGradientEval
export TProductSparseMatrixAssembler
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
export get_values
export get_index_map
include("TTArray.jl")
end # module
