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

using ROM.DofMaps
using ROM.TProduct
using ROM.ParamDataStructures
using ROM.ParamAlgebra

import Test: @test
import ArraysOfArrays: _innerlength
import PartitionedArrays: tuple_of_arrays
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction

export SingleFieldParamFESpace
include("ParamFESpaceInterface.jl")

export MultiFieldParamFESpace
include("MultiFieldParamFESpaces.jl")

export TrivialParamFESpace
include("TrivialParamFESpace.jl")

export TrialParamFESpace
export TrialParamFESpace!
export HomogeneousTrialParamFESpace
include("TrialParamFESpace.jl")

export ParamFEFunction
export SingleFieldParamFEFunction
export MultiFieldParamFEFunction
include("ParamFEFunction.jl")

end # module
