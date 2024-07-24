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
using Gridap.Geometry
using Gridap.CellData
using Gridap.Helpers

import Base:+,-,*,/,\
import Distributions: Uniform,Normal
import Statistics: mean
import Test: @test
import Gridap.Fields: BroadcastOpFieldArray,BroadcastingFieldOpMap,LinearCombinationField,LinearCombinationMap
import SparseArrays.getcolptr

import Mabla.FEM.IndexMaps: fast_index,slow_index,recast

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
export ParamSparseMatrix
export ParamArray
export param_array
export array_of_similar_arrays
export array_of_zero_arrays
export array_of_consecutive_arrays
export array_of_consecutive_zero_arrays
include("ParamArraysInterface.jl")

export ArrayOfArrays
export VectorOfVectors
export MatrixOfMatrices
include("ArrayOfArrays.jl")

export ConsecutiveArrayOfArrays
export ConsecutiveVectorOfVectors
export ConsecutiveMatrixOfMatrices
export AbstractConsecutiveParamVector
export AbstractConsecutiveParamMatrix
include("ConsecutiveArrayOfArrays.jl")

export MatrixOfSparseMatricesCSC
include("MatrixOfSparseMatrices.jl")

export VectorOfSparseVectors
include("VectorOfSparseVectors.jl")

export ConsecutiveParamArrays
export consecutive_getindex
export consecutive_setindex!
export consecutive_mul
include("ConsecutiveParamArrays.jl")

export ArrayOfTrivialArrays
include("ArrayOfTrivialArrays.jl")

export BlockArrayOfArrays
export BlockVectorOfVectors
export BlockMatrixOfMatrices
export BlockParamView
include("BlockArrayOfArrays.jl")

export ParamBroadcast
include("ParamBroadcasts.jl")

export ParamReindex
export PosNegParamReindex
include("ParamReindex.jl")

export Contribution
export ArrayContribution
export VectorContribution
export MatrixContribution
export TupOfArrayContribution
export contribution
export get_values
export get_parent
export order_triangulations
export find_closest_view
include("Contribution.jl")

# include("LazyMaps.jl")

end # module
