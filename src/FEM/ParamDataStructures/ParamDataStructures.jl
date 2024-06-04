module ParamDataStructures

using LinearAlgebra
using ArraysOfArrays
using BlockArrays
using ForwardDiff
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Arrays
using Gridap.Algebra
using Gridap.Fields
using Gridap.CellData
using Gridap.Helpers

import Base:+,-,*,/
import Distributions: Uniform,Normal
import Test: @test
import Gridap.Fields: BroadcastOpFieldArray,BroadcastingFieldOpMap,LinearCombinationField,LinearCombinationMap
import SparseArrays.getcolptr

export AbstractParamRealization
export ParamRealization
export TransientParamRealization
export UniformSampling
export NormalSampling
export ParamSpace
export TransientParamSpace
export ParamFunction, 𝑓ₚ
export TransientParamFunction, 𝑓ₚₜ
export realization
export get_params
export get_times
export num_params
export num_times
export slow_index
export fast_index
export shift!
include("ParamSpace.jl")

export ParamField
export ParamFieldGradient
export GenericParamField
export OperationParamField
include("ParamField.jl")

export AbstractParamContainer
export ParamContainer
include("ParamContainer.jl")

export ParamArray
export ParamVector
export ParamMatrix
export ParamSparseMatrix
export allocate_param_array
export zero_param_array
include("ParamArray.jl")

export MatrixOfSparseMatricesCSC
include("ArrayOfSparseMatrices.jl")

export ParamBlockArray
export ParamBlockVector
export ParamBlockMatrix
export ParamBlockArrayView
export ParamBlockVectorView
export ParamBlockMatrixView
include("ParamBlockArray.jl")

export Contribution
export ArrayContribution
export VectorContribution
export MatrixContribution
export TupOfArrayContribution
export contribution
export get_values
include("Contribution.jl")

export ParamReindex
export PosNegParamReindex
include("ParamReindex.jl")

end # module
