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

using ROManifolds.DofMaps
using ROManifolds.TProduct
using ROManifolds.ParamDataStructures
using ROManifolds.ParamAlgebra

import Test: @test
import ArraysOfArrays: _innerlength
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction

export collect_lazy_cell_matrix
export collect_lazy_cell_vector
export numeric_loop_lazy_matrix!
export numeric_loop_lazy_vector!
include("ParamAssemblers.jl")

export SingleFieldParamFESpace
export get_vector_type2
export param_zero_free_values
export param_zero_dirichlet_values
include("ParamFESpaceInterface.jl")

export MultiFieldParamFESpace
include("MultiFieldParamFESpaces.jl")

export TrivialParamFESpace
include("TrivialParamFESpaces.jl")

export TrialParamFESpace
export TrialParamFESpace!
export HomogeneousTrialParamFESpace
include("TrialParamFESpaces.jl")

export ParamFEFunction
export SingleFieldParamFEFunction
export MultiFieldParamFEFunction
include("ParamFEFunctions.jl")

export ParamFESolver
export LinearParamFESolver
export NonlinearParamFESolver
include("ParamFESolvers.jl")

end # module
