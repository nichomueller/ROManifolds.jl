module FEM
using Mabla.Utils

using LinearAlgebra
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

import Base: inv
import Base: abs
import Base: abs2
import Base: *
import Base: +
import Base: -
import Base: /
import Base: adjoint
import Base: transpose
import Base: real
import Base: imag
import Base: conj
import LinearAlgebra: det
import LinearAlgebra: tr
import LinearAlgebra: cross
import LinearAlgebra: dot
import LinearAlgebra: fillstored!
import BlockArrays: blockedrange
import FillArrays: Fill
import FillArrays: fill
import Distributions: Uniform
import Distributions: Normal
import ForwardDiff: derivative
import UnPack: @unpack
import Gridap.Helpers: @abstractmethod
import Gridap.Helpers: @check
import Gridap.Helpers: @notimplemented
import Gridap.Helpers: @unreachable
import Gridap.Algebra: InserterCSC
import Gridap.CellData: CellField
import Gridap.CellData: GenericMeasure
import Gridap.CellData: CompositeMeasure
import Gridap.CellData: DomainStyle
import Gridap.CellData: OperationCellField
import Gridap.CellData: change_domain
import Gridap.CellData: similar_cell_field
import Gridap.CellData: _get_cell_points
import Gridap.CellData: _operate_cellfields
import Gridap.CellData: _to_common_domain
import Gridap.Fields: OperationField
import Gridap.Fields: BroadcastOpFieldArray
import Gridap.Fields: BroadcastingFieldOpMap
import Gridap.Fields: LinearCombinationField
import Gridap.Fields: LinearCombinationMap
import Gridap.FESpaces: FEFunction
import Gridap.FESpaces: SparseMatrixAssembler
import Gridap.FESpaces: EvaluationFunction
import Gridap.FESpaces: _pair_contribution_when_possible
import Gridap.MultiField: MultiFieldFEBasisComponent
import Gridap.ReferenceFEs: get_order
import Gridap.ODEs.ODETools: residual!
import Gridap.ODEs.ODETools: jacobian!
import Gridap.ODEs.ODETools: jacobians!
import Gridap.ODEs.ODETools: _allocate_matrix_and_vector
import Gridap.ODEs.TransientFETools: ODESolver
import Gridap.ODEs.TransientFETools: ODEOperator
import Gridap.ODEs.TransientFETools: OperatorType
import Gridap.ODEs.TransientFETools: TransientCellField
import Gridap.ODEs.TransientFETools: TransientSingleFieldCellField
import Gridap.ODEs.TransientFETools: TransientMultiFieldCellField
import Gridap.ODEs.TransientFETools: TransientFEBasis
import Gridap.ODEs.TransientFETools: SingleFieldTypes
import Gridap.ODEs.TransientFETools: MultiFieldTypes
import Gridap.ODEs.TransientFETools: allocate_trial_space
import Gridap.ODEs.TransientFETools: fill_jacobians
import Gridap.ODEs.TransientFETools: _matdata_jacobian
import Gridap.ODEs.TransientFETools: _vcat_matdata
import Gridap.TensorValues: inner
import Gridap.TensorValues: outer
import Gridap.TensorValues: double_contraction
import Gridap.TensorValues: symmetric_part
import PartitionedArrays: tuple_of_arrays

export Table
export Affine

export ParametricSpace
export TransientParametricSpace
export UniformSampling
export NormalSampling
export Realization
export realization
include("ParametricSpace.jl")

export ∂ₚt
export ∂ₚtt
include("PDiffOperators.jl")

export PArray
export parray
export pzeros
export get_at_offsets
export recenter
include("PArray.jl")

export AbstractPTFunction
export PFunction
export PTFunction
export PField
export PGenericField
include("PField.jl")

export TrialPFESpace
export TrialPFESpace!
export HomogeneousTrialPFESpace
export MultiFieldPFESpace
export split_fields
export field_offsets
include("TrialPFESpace.jl")

export TransientTrialPFESpace
export TransientMultiFieldPFESpace
export TransientMultiFieldTrialPFESpace
include("TransientTrialPFESpace.jl")

export PCellField
export SingleFieldPFEFunction
export MultiFieldPFEFunction
include("PCellField.jl")

include("PAssemblers.jl")

export TransientPFEOperator
export TransientPFEOperatorFromWeakForm
export AffineTransientPFEOperator
export NonlinearTransientPFEOperator
export residual_for_trian!
export jacobian_for_trian!
include("TransientPFEOperator.jl")

export PODEOperator
export ConstantPODEOperator
export ConstantMatrixPODEOperator
export AffinePODEOperator
export NonlinearPODEOperator
export PODEOpFromFEOp
include("PODEOperatorInterface.jl")

export PODESolution
export num_time_dofs
export get_times
export get_stencil_times
export _check_convergence
include("PTSolvers.jl")

export PThetaMethodOperator
export AffinePThetaMethodOperator
include("PThetaMethod.jl")

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
