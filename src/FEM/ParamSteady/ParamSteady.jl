module ParamSteady

using LinearAlgebra
using BlockArrays
using SparseArrays

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.Fields
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Helpers

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers
using GridapSolvers.MultilevelTools

using ReducedOrderModels.IndexMaps
using ReducedOrderModels.TProduct
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamAlgebra
using ReducedOrderModels.ParamFESpaces

import Test: @test
import UnPack: @unpack
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ReferenceFEs: get_order
import ReducedOrderModels.Utils: CostTracker

export UnEvalParamSingleFieldFESpace
export ParamTrialFESpace
export ParamMultiFieldFESpace
include("ParamTrialFESpace.jl")

export UnEvalOperatorType
export NonlinearParamEq
export LinearParamEq
export LinearNonlinearParamEq
export ParamOperator
export LinearNonlinearParamOpFromFEOp
export AbstractParamCache
export ParamCache
export LinearNonlinearParamCache
export ParamNonlinearOperator
export allocate_paramcache
export update_paramcache!
export get_fe_operator
include("ParamOperator.jl")

export ParamFEOperator
export LinearParamFEOperator
export get_param_space
include("ParamFEOperator.jl")

export ParamFEOperatorWithTrian
export set_triangulation
export change_triangulation
include("ParamFEOperatorWithTrian.jl")

export LinearNonlinearParamFEOperator
export get_linear_operator
export get_nonlinear_operator
export join_operators
include("LinearNonlinearParamFEOperator.jl")

export ParamOpFromFEOp
export ParamOpFromFEOpWithTrian
include("ParamOpFromFEOp.jl")

include("ParamSolutions.jl")

end # module
