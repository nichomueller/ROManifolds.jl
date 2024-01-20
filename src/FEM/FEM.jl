module FEM
using Mabla.Utils

using LinearAlgebra
using BlockArrays
using SparseArrays
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

export Table
export Affine

export PRealization
export TransientPRealization
export UniformSampling
export NormalSampling
export ParametricSpace
export TransientParametricSpace
export PFunction, 𝑓ₚ
export TransientPFunction, 𝑓ₚₜ
export realization
export get_parameters
export get_times
export get_fields
include("ParametricSpace.jl")

include("PDiffOperators.jl")

export PArray
export allocate_parray
export zero_parray
export get_at_offsets
export recenter
include("PArray.jl")

export PField
export PFieldGradient
export GenericPField
export ZeroPField
export ConstantPField
export OperationPField
include("PField.jl")

export PCellField
export SingleFieldPFEFunction
export MultiFieldPFEFunction
include("PCellField.jl")

export TrialPFESpace
export TrialPFESpace!
export HomogeneousTrialPFESpace
export length_dirichlet_values
export length_free_values
include("TrialPFESpace.jl")

export MultiFieldPFESpace
export MultiFieldPFEFunction
include("MultiFieldPFESpaces.jl")

export TransientTrialPFESpace
export TransientMultiFieldPFESpace
export TransientMultiFieldTrialPFESpace
include("TransientTrialPFESpace.jl")

export SparseMatrixPAssembler
include("PAssemblers.jl")

export TransientPFEOperator
export TransientPFEOperatorFromWeakForm
export AffineTransientPFEOperator
export NonlinearTransientPFEOperator
export residual_for_trian!
export jacobian_for_trian!
include("TransientPFEOperator.jl")

export ODEPOperator
export ConstantODEPOperator
export ConstantMatrixODEPOperator
export AffineODEPOperator
export NonlinearODEPOperator
export ODEPOpFromFEOp
include("ODEPOperatorInterface.jl")

export PThetaMethodOperator
export AffinePThetaMethodOperator
include("PThetaMethod.jl")

export ODEPSolution
export GenericODEPSolution
include("ODEPSolution.jl")

export TransientPFESolution
include("TransientPFESolution.jl")

export PString
export PVisualizationData
include("PVisualization.jl")

export get_order
export get_L2_norm_matrix
export get_H1_norm_matrix
include("FEUtils.jl")

export ReducedMeasure
include("ReducedMeasure.jl")
end # module
