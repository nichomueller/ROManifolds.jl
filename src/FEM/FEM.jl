module FEM
using Mabla.Utils

using LinearAlgebra
using BlockArrays
using SparseArrays
using SparseMatricesCSR
using WriteVTK
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools
using Gridap.Visualization
using Gridap.Helpers

import Base: inv,abs,abs2,*,+,-,/,adjoint,transpose,real,imag,conj
import LinearAlgebra: det,tr,cross,dot,fillstored!
import FillArrays: Fill,fill
import ForwardDiff: derivative
import Distributions: Uniform,Normal
import Test: @test
import UnPack: @unpack
import Gridap.Algebra: residual!,jacobian!
import Gridap.Fields: OperationField,BroadcastOpFieldArray,BroadcastingFieldOpMap,LinearCombinationField,LinearCombinationMap
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ODEs.ODETools: jacobians!
import Gridap.ODEs.TransientFETools: TransientCellField,allocate_trial_space,fill_jacobians,_matdata_jacobian
import Gridap.ReferenceFEs: get_order
import Gridap.TensorValues: inner,outer,double_contraction,symmetric_part
import PartitionedArrays: tuple_of_arrays

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
include("ParamSpace.jl")

include("DiffOperators.jl")

export AbstractParamContainer
export ParamContainer
include("ParamContainer.jl")

export ParamArray
export ParamVector
export ParamMatrix
export allocate_param_array
export zero_param_array
export get_at_offsets
export recenter
include("ParamArray.jl")

export ParamBlockArray
include("ParamBlockArray.jl")

export ParamReindex
export PosNegParamReindex
include("ParamReindex.jl")

export ParamField
export ParamFieldGradient
export GenericParamField
export ZeroParamField
export ConstantParamField
export OperationParamField
include("ParamField.jl")

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
export TransientMultiFieldTrialParamFESpace
include("TransientTrialParamFESpace.jl")

export ParamCounter
export ParamInserterCSC
export ParamCSSR
include("ParamAlgebra.jl")

include("ParamAssemblers.jl")

include("ParamBlockAssemblers.jl")

export TransientParamFEOperator
export TransientParamFEOperatorFromWeakForm
export AffineTransientParamFEOperator
export NonlinearTransientParamFEOperator
export residual_for_trian!
export jacobian_for_trian!
include("TransientParamFEOperator.jl")

export ODEParamOperator
export ConstantODEParamOperator
export ConstantMatrixODEParamOperator
export AffineODEParamOperator
export NonlinearODEParamOperator
export ODEParamOpFromFEOp
include("ODEParamOperatorInterface.jl")

export ThetaMethodParamOperator
export AffineThetaMethodParamOperator
include("ThetaMethod.jl")

export ODEParamSolution
export GenericODEParamSolution
include("ODEParamSolution.jl")

export TransientParamFESolution
include("TransientParamFESolution.jl")

export ParamString
export ParamVisualizationData
include("ParamVisualization.jl")

export get_order
export get_L2_norm_matrix
export get_H1_norm_matrix
include("FEUtils.jl")
end # module
