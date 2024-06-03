module ParamFESpaces

using LinearAlgebra
using BlockArrays
using SparseArrays

using Gridap
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Helpers

using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra

import Test: @test
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction

export SingleFieldParamFESpace
export FESpaceToParamFESpace
export length_dirichlet_values
export length_free_values
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

include("ParamAssemblers.jl")

include("ParamBlockAssemblers.jl")

end # module
