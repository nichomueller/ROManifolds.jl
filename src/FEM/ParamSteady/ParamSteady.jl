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

using Mabla.FEM.IndexMaps
using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures
using Mabla.FEM.ParamAlgebra
using Mabla.FEM.ParamFESpaces

import Test: @test
import UnPack: @unpack
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ReferenceFEs: get_order

export ParametricSingleFieldFESpace
export ParamTrialFESpace
export ParamMultiFieldFESpace
include("ParamTrialFESpace.jl")

export ParamOperatorType
export NonlinearParamEq
export LinearParamEq
export ParamOperator
export ParamOperatorWithTrian
export ParamOpFromFEOpCache
export allocate_paramcache
export update_paramcache!
include("ParamOperator.jl")

export ParamFEOperator
export LinearParamFEOperator
export ParamSaddlePointFEOp
export ParamFEOperatorWithTrian
export ParamFEOpFromWeakFormWithTrian
export ParamSaddlePointFEOpWithTrian
export assemble_norm_matrix
export assemble_coupling_matrix
export set_triangulation
export change_triangulation
include("ParamFEOperator.jl")

export LinearNonlinearParamODE
export ParamLinNonlinFEOperator
export ParamLinearNonlinearFEOperator
export get_linear_operator
export get_nonlinear_operator
export join_operators
include("ParamLinearNonlinearFEOperator.jl")

export ParamOpFromFEOp
export ParamOpFromFEOpWithTrian
include("ParamOpFromFEOp.jl")

end # module
