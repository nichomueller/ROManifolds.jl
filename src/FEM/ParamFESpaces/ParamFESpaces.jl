module ParamFESpaces

using LinearAlgebra
using BlockArrays
using SparseArrays

using Gridap
using Gridap.Algebra
using Gridap.CellData
using Gridap.Fields
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Helpers

using Mabla.FEM.ParamDataStructures
using Mabla.FEM.IndexMaps
using Mabla.FEM.ParamAlgebra

import Test: @test
import ArraysOfArrays: _innerlength
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction

export SingleFieldParamFESpace
export FESpaceToParamFESpace
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
include("ParamAssemblers.jl")

include("ParamBlockAssemblers.jl")

end # module
