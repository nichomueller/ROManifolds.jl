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

export FEOperatorIndexMap
export get_vector_index_map
export get_matrix_index_map
include("FEOperatorIndexMaps.jl")

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
export ParamOpFromFEOpCache
export allocate_paramcache
export update_paramcache!
include("ParamOperator.jl")

export ParamFEOperator
export LinearParamFEOperator
export ParamFEOperatorWithTrian
export ParamFEOpFromWeakFormWithTrian
export assemble_matrix_from_form
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

end # module
