module TProduct
using LinearAlgebra
using SparseArrays
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

include("Utils.jl")

export IndexMap
export NodesMap
export compute_nodes_map
include("IndexMaps.jl")

export Isotropy
export Isotropic
export Anisotropic
export TensorProductNodes
export get_factors
export get_indices_map
export get_isotropy
include("TProductNodes.jl")

export TensorProductMap
export TensorProductField
export GenericTensorProductField
include("TProductFields.jl")

export TensorProductMonomialBasis
export ParamContainer
include("TProductMonomial.jl")

export TensorProductDofBases
include("TProductBasis.jl")

export TensorProductShapefuns
include("TProductShapefuns.jl")

export TensorProductArray
include("TProductArray.jl")

export TensorProductRefFE
export tplagrangian
include("TProductReffe.jl")

export TensorProductQuadrature
export tpquadrature
include("TProductQuadrature.jl")

export TensorProductDescriptor
export KroneckerCoordinates
export TensorProductGrid
include("TProductGeometry.jl")

# export TProductFESpace
export get_dof_permutation
export comp_to_free_dofs
include("TProductFESpaces.jl")

export TProductCellFields
export TProductFESpace
export TProductFEBasisComponent
export TProductGradient
include("TProductCellFields.jl")
end # module
