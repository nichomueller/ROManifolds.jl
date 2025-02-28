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

using ROManifolds.Utils
using ROManifolds.DofMaps

import Base:+,-,*,/,\
import Distributions: Uniform,Normal
import FillArrays: Fill
import HaltonSequences: HaltonPoint
import LatinHypercubeSampling: randomLHC,scaleLHC
import StatsBase: sample
import Test: @test
import Gridap.Fields: BroadcastOpFieldArray,BroadcastingFieldOpMap,LinearCombinationField,LinearCombinationMap
import SparseArrays.getcolptr

export AbstractRealization
export Realization
export TransientRealization
export UniformSampling
export NormalSampling
export HaltonSampling
export ParamSpace
export TransientParamSpace
export AbstractParamFunction
export ParamFunction
export TransientParamFunction
export realization
export parameterize
export get_params
export get_times
export get_at_time
export num_params
export num_times
export get_initial_time
export get_final_time
export shift!
include("ParamSpaces.jl")

export AbstractParamContainer
export get_param_data
export eltype2
export param_length
export param_eachindex
export param_getindex
export param_setindex!
export get_param_entry
export get_param_entry!
include("ParamContainersInterface.jl")

export AbstractParamArray
export AbstractParamVector
export AbstractParamMatrix
export ParamArray
export ParamVector
export ParamMatrix
export MemoryLayoutStyle
export ConsecutiveMemory
export NonConsecutiveMemory
export consecutive_parameterize
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
include("BlockParamArrays.jl")

export ParamVectorWithEntryRemoved
export ParamVectorWithEntryInserted
include("ParamVectorWithEntries.jl")

export AbstractSnapshots
export Snapshots
export SteadySnapshots
export GenericSnapshots
export SnapshotsAtIndices
export SparseSnapshots
export BlockSnapshots
export get_realization
export select_snapshots
export get_indexed_data
include("Snapshots.jl")

export TransientSnapshots
export TransientGenericSnapshots
export TransientSparseSnapshots
export TransientSnapshotsWithIC
export UnfoldingTransientSnapshots
export ModeTransientSnapshots
export get_initial_data
export change_mode
include("TransientSnapshots.jl")

include("ParamBroadcasts.jl")

export ParamField
export TrivialParamField
export GenericParamField
include("ParamFields.jl")

include("ParamReindex.jl")

end # module
