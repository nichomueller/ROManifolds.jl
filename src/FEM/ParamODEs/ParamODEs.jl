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
using Gridap.Helpers

using ROM.Utils
using ROM.DofMaps
using ROM.TProduct
using ROM.ParamDataStructures
using ROM.ParamFESpaces
using ROM.ParamSteady

import BlockArrays: blocks
import Test: @test
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ODEs: TransientCellField
import Gridap.ReferenceFEs: get_order
import ROM.ParamSteady: get_domains_res,get_domains_jac

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
include("TransientParamFEOperator.jl")

export LinearNonlinearParamODE
export LinearNonlinearTransientParamFEOperator
include("LinearNonlinearTransientParamFEOperator.jl")

export ODEParamOpFromTFEOp
include("ODEParamOpFromTFEOp.jl")

include("ThetaMethod.jl")

export collect_initial_values
include("ODEParamSolution.jl")

include("TransientParamFESolution.jl")

end # module
