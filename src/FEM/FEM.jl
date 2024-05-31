module FEM
using LinearAlgebra
using BlockArrays
using SparseArrays
using SparseMatricesCSR
using ForwardDiff
using WriteVTK
using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Polynomials
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Visualization
using Gridap.Helpers
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers
using GridapSolvers.MultilevelTools

import Base: inv,abs,abs2,*,+,-,/,adjoint,transpose,real,imag,conj
import LinearAlgebra: det,tr,cross,dot,fillstored!
import FillArrays: Fill,fill
import Distributions: Uniform,Normal
import Kronecker: kronecker
import Test: @test
import UnPack: @unpack
import Gridap.Algebra: residual!,jacobian!
import Gridap.Fields: OperationField,BroadcastOpFieldArray,BroadcastingFieldOpMap,LinearCombinationField,LinearCombinationMap
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ODEs: TransientCellField
import Gridap.ReferenceFEs: get_order
import Gridap.TensorValues: Mutable,inner,outer,double_contraction,symmetric_part
import PartitionedArrays: tuple_of_arrays
import SparseArrays: AbstractSparseMatrixCSC

export AbstractParamRealization
export ParamRealization
export TransientParamRealization
export UniformSampling
export NormalSampling
export ParamSpace
export TransientParamSpace
export ParamFunction, 𝑓ₚ
export TransientParamFunction, 𝑓ₚₜ
export realization
export get_params
export get_times
export num_params
export num_times
export slow_index
export fast_index
export shift!
include("ParamSpace.jl")

include("TimeDerivatives.jl")

export ParamField
export ParamFieldGradient
export GenericParamField
export ZeroParamField
export ConstantParamField
export OperationParamField
include("ParamField.jl")

export AbstractParamContainer
export ParamContainer
include("ParamContainer.jl")

export ParamArray
export ParamVector
export ParamMatrix
export ParamSparseMatrix
export allocate_param_array
export zero_param_array
include("ParamArray.jl")

export ParamBlockArray
export ParamBlockVector
include("ParamBlockArray.jl")

export Contribution
export ArrayContribution
export TupOfArrayContribution
export contribution
export get_values
include("Contribution.jl")

export ParamReindex
export PosNegParamReindex
include("ParamReindex.jl")

include("LagrangianDofBases.jl")

export SingleFieldParamFESpace
export FESpaceToParamFESpace
export length_dirichlet_values
export length_free_values
include("ParamFESpaceInterface.jl")

export MultiFieldParamFESpace
include("MultiFieldParamFESpaces.jl")

export TrialParamFESpace
export TrialParamFESpace!
export HomogeneousTrialParamFESpace
include("TrialParamFESpace.jl")

export ParamFEFunction
export SingleFieldParamFEFunction
export MultiFieldParamFEFunction
include("ParamFEFunction.jl")

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace
include("TransientTrialParamFESpace.jl")

export ParamCounter
export ParamInserterCSC
export ParamCSSR
include("ParamAlgebra.jl")

include("ParamAssemblers.jl")

include("ParamBlockAssemblers.jl")

include("FastLinearSolvers.jl")

include("ParamIterativeSolvers.jl")

export ODEParamOperatorType
export NonlinearParamODE
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
export assemble_norm_matrix
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

export get_parent
include("TriangulationParents.jl")

export SparsityPattern
export SparsityPatternCSC
export TProductSparsityPattern
export get_sparsity
include("SparsityPatterns.jl")

export AbstractIndexMap
export IndexMap
export IndexMapView
export MultiValueIndexMap
export FixedDofIndexMap
export TProductIndexMap
export SparseIndexMap
export inv_index_map
export free_dofs_map
export vectorize_index_map
export recast_indices
export sparsify_indices
export get_nonzero_indices
export tensorize_indices
export split_row_col_indices
include("TProductIndexMaps.jl")

export TProductModel
export TProductTriangulation
export TProductMeasure
include("TProductGeometry.jl")

export TProductFESpace
export TProductFEBasis
export get_dof_permutation
export get_sparse_index_map
export get_tp_dof_permutation
export get_tp_fe_basis
export get_tp_trial_fe_basis
include("TProductFESpaces.jl")

export TProductCellPoint
export TProductCellFields
export TProductGradientCellField
export TProductGradientEval
export TProductSparseMatrixAssembler
export TProductArray
export TProductGradientArray
export symbolic_kron
export symbolic_kron!
export numerical_kron!
export kronecker_gradients
include("TProductCellFields.jl")

export TTArray
export TTVector
export TTMatrix
export ParamTTArray
export ParamTTVector
export ParamTTMatrix
export get_values
export get_index_map
include("TTArray.jl")

export TTBuilder
export TTCounter
export TTInserter
export ParamTTInserterCSC
include("TTAlgebra.jl")

export TTBlockArray
export TTBlockVector
export TTBlockMatrix
export ParamBlockTTArray
export ParamBlockTTVector
export ParamBlockTTMatrix
include("TTBlockArray.jl")
end # module
