module FEM
using Mabla.Utils

using LinearAlgebra
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
import Distributions: Uniform
import Distributions: Normal
import ForwardDiff: derivative
import FillArrays: Fill
import FillArrays: fill
import BlockArrays: blockedrange
import UnPack: @unpack
import Gridap.Helpers: @abstractmethod
import Gridap.Helpers: @check
import Gridap.Helpers: @notimplemented
import Gridap.Helpers: @unreachable
import Gridap.Algebra: InserterCSC
import Gridap.Algebra: LUNumericalSetup
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

export Table
export Affine

export PSpace
export UniformSampling
export NormalSampling
export realization
include("PSpace.jl")

export ∂ₚt
export ∂ₚtt
include("PDiffOperators.jl")

export PTArray
export ptzeros
export get_at_offsets
export recenter
include("PTArray.jl")

export AbstractPTFunction
export PFunction
export PTFunction
export PTField
export PTGenericField
include("PTFields.jl")

export TrialPFESpace
export TrialPFESpace!
export HomogeneousTrialPFESpace
export MultiFieldPFESpace
export split_fields
export field_offsets
include("PFESpaces.jl")

export PTCellField
export SingleFieldPTFEFunction
export MultiFieldPTFEFunction
include("PTCellFields.jl")

export PTIntegrand
export ∫ₚ
export CollectionPTIntegrand
include("PTIntegrand.jl")

include("PTAssemblers.jl")

export TransientTrialPFESpace
export TransientMultiFieldPFESpace
export TransientMultiFieldTrialPFESpace
include("PTFESpaces.jl")

export PTFEOperator
export PTFEOperatorFromWeakForm
export AffinePTFEOperator
export NonlinearPTFEOperator
export residual_for_trian!
export jacobian_for_trian!
include("PTFEOperator.jl")

export PTNonlinearOperator
export PTAlgebraicOperator
export update_algebraic_operator
include("PTAlgebraicOperator.jl")

export PODESolver
export PThetaMethod
export PODESolution
export num_time_dofs
export get_times
export _check_convergence
include("PTSolvers.jl")

export PTThetaMethodOperator
export PTAffineThetaMethodOperator
include("PThetaMethod.jl")

export get_order
export get_L2_norm_matrix
export get_H1_norm_matrix
include("FEUtils.jl")

export ReducedMeasure
include("ReducedMeasure.jl")
end # module
