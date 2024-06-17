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
using Gridap.ReferenceFEs
using Gridap.Helpers

using Mabla.FEM.IndexMaps
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamFESpaces
using Mabla.FEM.ParamSteady

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
export QuasilinearParamODE
export SemilinearParamODE
export LinearParamODE
export ODEParamOperator
export ODEParamOperatorWithTrian
export ParamODEOpFromTFEOpCache
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

export TransientParamSaddlePointFEOp
export assemble_coupling_matrix
include("TransientParamSaddlePointFEOperator.jl")

export TransientParamFEOperatorWithTrian
export TransientParamFEOpFromWeakFormWithTrian
export TransientParamSaddlePointFEOpWithTrian
export set_triangulation
export change_triangulation
include("TransientParamFEOperatorWithTrian.jl")

export LinearNonlinearParamODE
export TransientParamLinNonlinFEOperator
export TransientParamLinearNonlinearFEOperator
export get_linear_operator
export get_nonlinear_operator
export join_operators
include("TransientParamLinearNonlinearFEOperator.jl")

export ODEParamOpFromTFEOp
export ODEParamOpFromTFEOpWithTrian
include("ODEParamOpFromTFEOp.jl")

include("ThetaMethod.jl")

export ODEParamSolution
export GenericODEParamSolution
include("ODEParamSolution.jl")

export TransientParamFESolution
include("TransientParamFESolution.jl")

end # module
