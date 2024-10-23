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

include("TimeDerivatives.jl")

include("TransientParamCellField.jl")

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace
include("TransientTrialParamFESpace.jl")

export ODEParamOperatorType
export NonlinearParamODE
export AbstractLinearParamODE
export SemilinearParamODE
export LinearParamODE
export ODEParamOperator
include("ODEParamOperator.jl")

export ParamStageOperator
export NonlinearParamStageOperator
export LinearParamStageOperator
include("ParamStageOperator.jl")

export TransientParamFEOperator
export TransientParamFEOpFromWeakForm
export TransientParamSemilinearFEOperator
export TransientParamSemilinearFEOpFromWeakForm
export TransientParamLinearFEOperator
export TransientParamLinearFEOpFromWeakForm
export LinearTransientParamFEOperator
export NonlinearTransientParamFEOperator
include("TransientParamFEOperator.jl")

export TransientParamFEOperatorWithTrian
export TransientParamFEOperatorWithTrian
export set_triangulation
export change_triangulation
include("TransientParamFEOperatorWithTrian.jl")

export LinearNonlinearParamODE
export LinearNonlinearTransientParamFEOperator
export get_linear_operator
export get_nonlinear_operator
export join_operators
include("LinearNonlinearTransientParamFEOperator.jl")

export ODEParamOpFromTFEOp
export ODEParamOpFromTFEOpWithTrian
include("ODEParamOpFromTFEOp.jl")

include("ThetaMethod.jl")

export ODEParamSolution
include("ODEParamSolution.jl")

export TransientParamFESolution
include("TransientParamFESolution.jl")

end # module
