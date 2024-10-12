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

export ParamOperatorType
export NonlinearParamEq
export LinearParamEq
export LinearNonlinearParamEq
export ParamOperator
export ParamOperatorWithTrian
export ParamCache
export allocate_paramcache
export update_paramcache!
include("ParamOperator.jl")

export ParamFEOperator
export LinearParamFEOperator
export ParamFEOperatorWithTrian
export ParamFEOpFromWeakFormWithTrian
export set_triangulation
export change_triangulation
include("ParamFEOperator.jl")

export LinearNonlinearParamODE
export LinNonlinParamFEOperator
export LinearNonlinearParamFEOperator
export LinearNonlinearParamFEOperatorWithTrian
export get_linear_operator
export get_nonlinear_operator
export join_operators
include("LinearNonlinearParamFEOperator.jl")

export ParamOpFromFEOp
export ParamOpFromFEOpWithTrian
include("ParamOpFromFEOp.jl")

include("ParamSolutions.jl")

end # module
