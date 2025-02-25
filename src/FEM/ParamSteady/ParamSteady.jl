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

using ROManifolds.Utils
using ROManifolds.DofMaps
using ROManifolds.TProduct
using ROManifolds.ParamDataStructures
using ROManifolds.ParamAlgebra
using ROManifolds.ParamFESpaces

import Test: @test
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ReferenceFEs: get_order
import ROManifolds.Utils: CostTracker

export UnEvalTrialFESpace
export ParamTrialFESpace
include("ParamTrialFESpaces.jl")

export UnEvalOperatorType
export NonlinearParamEq
export LinearParamEq
export LinearNonlinearParamEq
export TriangulationStyle
export SplitDomains
export JointDomains
export ParamOperator
export LinearNonlinearParamOpFromFEOp
export AbstractParamCache
export ParamOpCache
export ParamOpSysCache
export ParamNonlinearOperator
export allocate_paramcache
export update_paramcache!
export allocate_systemcache
export get_fe_operator
include("ParamOperators.jl")

export ParamFEOperator
export SplitParamFEOperator
export JointParamFEOperator
export LinearParamFEOperator
export FEDomains
export get_param_space
export get_param_assembler
export get_domains
export get_domains_res
export get_domains_jac
export set_domains
export change_domains
export get_dof_map_at_domains
export get_sparse_dof_map_at_domains
include("ParamFEOperators.jl")

export LinearNonlinearParamFEOperator
export get_linear_operator
export get_nonlinear_operator
export join_operators
include("LinearNonlinearParamFEOperators.jl")

export ParamOpFromFEOp
export JointParamOpFromFEOp
export SplitParamOpFromFEOp
export collect_cell_matrix_for_trian
export collect_cell_vector_for_trian
include("ParamOpFromFEOps.jl")

include("ParamSolutions.jl")

end # module
