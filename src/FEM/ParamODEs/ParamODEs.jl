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

using ROManifolds.Utils
using ROManifolds.DofMaps
using ROManifolds.TProduct
using ROManifolds.ParamDataStructures
using ROManifolds.ParamAlgebra
using ROManifolds.ParamFESpaces
using ROManifolds.ParamSteady

import BlockArrays: blocks
import Test: @test
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ODEs: TransientCellField
import Gridap.ReferenceFEs: get_order
import ROManifolds.ParamSteady: get_domains_res,get_domains_jac

include("TimeDerivatives.jl")

include("TransientParamCellFields.jl")

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace
include("TransientTrialParamFESpaces.jl")

export ODEParamOperatorType
export NonlinearParamODE
export LinearParamODE
export ODEParamOperator
include("ODEParamOperators.jl")

export ParamStageOperator
include("ParamStageOperators.jl")

export TransientParamFEOperator
export SplitTransientParamFEOperator
export JointTransientParamFEOperator
export TransientParamFEOpFromWeakForm
export TransientParamLinearFEOperator
export TransientParamLinearFEOpFromWeakForm
include("TransientParamFEOperators.jl")

export LinearNonlinearParamODE
export LinearNonlinearTransientParamFEOperator
include("LinearNonlinearTransientParamFEOperators.jl")

export ODEParamOpFromTFEOp
include("ODEParamOpFromTFEOps.jl")

include("ThetaMethod.jl")

export collect_initial_values
include("ODEParamSolutions.jl")

include("TransientParamFESolutions.jl")

end # module
