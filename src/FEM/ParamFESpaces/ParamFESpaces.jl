module ParamFESpaces

using LinearAlgebra
using BlockArrays
using FillArrays
using SparseArrays

using Gridap
using Gridap.Arrays
using Gridap.Algebra
using Gridap.CellData
using Gridap.Fields
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Helpers

using ReducedOrderModels.IndexMaps
using ReducedOrderModels.TProduct
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamAlgebra

import Test: @test
import ArraysOfArrays: _innerlength
import PartitionedArrays: tuple_of_arrays
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction

export SingleFieldParamFESpace
export TrivialParamFESpace
include("ParamFESpaceInterface.jl")

export MultiFieldParamFESpace
include("MultiFieldParamFESpaces.jl")

export TrialParamFESpace
export TrialParamFESpace!
export HomogeneousTrialParamFESpace
include("TrialParamFESpace.jl")

export ParamFEFunction
export SingleFieldParamFEFunction
export MultiFieldParamFEFunction
include("ParamFEFunction.jl")

export get_param_assembler
export collect_cell_matrix_for_trian
export collect_cell_vector_for_trian
include("ParamAssemblers.jl")

include("ParamBlockAssemblers.jl")

include("LagrangianDofBases.jl")

end # module
