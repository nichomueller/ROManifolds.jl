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

export ParamContainer
include("ParamContainers.jl")

export MatrixOfMatrices
include("MatrixOfMatrices.jl")

export MatrixOfSparseMatricesCSC
include("MatrixOfSparseMatrices.jl")

export ParamArray
export ParamVector
export ParamMatrix
export param_data
export param_length
export param_eachindex
export param_getindex
include("ParamArrays.jl")

export ParamBroadcast
include("ParamBroadcasts.jl")

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
