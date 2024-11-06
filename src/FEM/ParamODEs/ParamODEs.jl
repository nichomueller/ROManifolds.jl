module ParamODEs

using LinearAlgebra
using ForwardDiff

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Polynomials
using Gridap.Helpers

using ReducedOrderModels.Utils
using ReducedOrderModels.IndexMaps
using ReducedOrderModels.TProduct
using ReducedOrderModels.ParamDataStructures
using ReducedOrderModels.ParamFESpaces
using ReducedOrderModels.ParamSteady

import Test: @test
import UnPack: @unpack
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ODEs: TransientCellField
import Gridap.ReferenceFEs: get_order
import ReducedOrderModels.ParamSteady: get_trian_res,get_trian_jac

include("TimeDerivatives.jl")

include("TransientParamCellField.jl")

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace
include("TransientTrialParamFESpace.jl")

export ODEParamOperatorType
export NonlinearParamODE
export LinearParamODE
export ODEParamOperator
include("ODEParamOperator.jl")

export ParamStageOperator
include("ParamStageOperator.jl")

export TransientParamFEOperator
export SplitTransientParamFEOperator
export JointTransientParamFEOperator
export TransientParamFEOpFromWeakForm
export TransientParamLinearFEOperator
export TransientParamLinearFEOpFromWeakForm
export LinearTransientParamFEOperator
export NonlinearTransientParamFEOperator
include("TransientParamFEOperator.jl")

export LinearNonlinearParamODE
export LinearNonlinearTransientParamFEOperator
export get_linear_operator
export get_nonlinear_operator
export join_operators
include("LinearNonlinearTransientParamFEOperator.jl")

export ODEParamOpFromTFEOp
include("ODEParamOpFromTFEOp.jl")

include("ThetaMethod.jl")

export ODEParamSolution
include("ODEParamSolution.jl")

export TransientParamFESolution
include("TransientParamFESolution.jl")

end # module
