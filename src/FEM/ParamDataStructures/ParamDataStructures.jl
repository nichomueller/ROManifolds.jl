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

import Base:+,-,*,/,\
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
export AbstractParamFunction
export ParamFunction, 𝑓ₚ
export TransientParamFunction, 𝑓ₚₜ
export realization
export get_params
export get_times
export get_initial_time
export get_final_time
export num_params
export num_times
export slow_index
export fast_index
export shift!
include("ParamSpace.jl")

export AbstractParamContainer
export ParamContainer
export ParamNumber
export param_data
export param_length
export param_eachindex
export param_getindex
export param_setindex!
export param_view
export param_entry
include("ParamContainersInterface.jl")

export ParamField
export ParamFieldGradient
export GenericParamField
export OperationParamField
include("ParamField.jl")

export AbstractParamArray
export AbstractParamVector
export AbstractParamMatrix
export ParamArray
export param_array
export array_of_similar_arrays
export array_of_zero_arrays
include("ParamArraysInterface.jl")

export ArrayOfArrays
export VectorOfVectors
export MatrixOfMatrices
include("ArrayOfArrays.jl")

export ArrayOfCachedArrays
export VectorOfCachedVectors
export MatrixOfCachedMatrices
include("ArrayOfCachedArrays.jl")

export MatrixOfSparseMatricesCSC
include("MatrixOfSparseMatrices.jl")

export ArrayOfTrivialArrays
include("ArrayOfTrivialArrays.jl")

export BlockArrayOfArrays
export BlockVectorOfVectors
export BlockMatrixOfMatrices
export BlockParamView
include("BlockArrayOfArrays.jl")

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
