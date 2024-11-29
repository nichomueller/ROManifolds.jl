module ParamDataStructures

using LinearAlgebra
using ArraysOfArrays
using BlockArrays
using ForwardDiff
using HaltonSequences
using SmolyakApprox
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
import FillArrays: Fill
import Statistics: mean
import Test: @test
import Gridap.Fields: BroadcastOpFieldArray,BroadcastingFieldOpMap,LinearCombinationField,LinearCombinationMap
import SparseArrays.getcolptr

import ReducedOrderModels.DofMaps: fast_index,slow_index,recast

export AbstractRealization
export Realization
export TransientRealization
export UniformSampling
export NormalSampling
export HaltonSampling
export SmolyakSampling
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
export shift!
include("ParamSpace.jl")

export AbstractParamContainer
export ParamType
export PType
export ParamContainer
export ParamNumber
export get_param_data
export param_length
export param_eachindex
export param_getindex
export param_setindex!
export get_param_entry
export get_param_entry!
export param_typeof
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
export ParamVector
export ParamMatrix
export MemoryLayoutStyle
export ConsecutiveMemory
export NonConsecutiveMemory
export param_array
export consecutive_param_array
export innersize
export innerlength
export inneraxes
include("ParamArraysInterface.jl")

export TrivialParamArray
export ConsecutiveParamArray
export ConsecutiveParamVector
export ConsecutiveParamMatrix
export GenericParamVector
export GenericParamMatrix
export ArrayOfArrays
export get_all_data
include("ParamArrays.jl")

export ParamSparseMatrix
export ParamSparseMatrixCSC
export ParamSparseMatrixCSR
export ConsecutiveParamSparseMatrixCSC
export GenericParamSparseMatrixCSC
export ConsecutiveParamSparseMatrixCSR
export GenericParamSparseMatrixCSR
export ConsecutiveParamSparseMatrix
include("ParamSparseMatrices.jl")

export BlockParamArray
export BlockParamVector
export BlockParamMatrix
export BlockConsecutiveParamVector
export BlockConsecutiveParamMatrix
export nblocks
include("BlockParamArrays.jl")

export ParamVectorWithEntryRemoved
export ParamVectorWithEntryInserted
include("ParamVectorWithEntries.jl")

include("ParamBroadcasts.jl")

export ParamReindex
export PosNegParamReindex
include("ParamReindex.jl")

end # module
