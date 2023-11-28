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
import LinearAlgebra: ⋅
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
import Gridap.FESpaces: FEFunction
import Gridap.FESpaces: SparseMatrixAssembler
import Gridap.FESpaces: EvaluationFunction
import Gridap.FESpaces: _pair_contribution_when_possible
import Gridap.MultiField: MultiFieldFEBasisComponent
import Gridap.ReferenceFEs: get_order
import Gridap.ODEs.ODETools: _allocate_matrix_and_vector
import Gridap.ODEs.ODETools: residual!
import Gridap.ODEs.ODETools: jacobian!
import Gridap.ODEs.ODETools: jacobians!
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
import Gridap.ODEs.TransientFETools: _to_transient_single_fields
import Gridap.TensorValues: inner
import Gridap.TensorValues: outer
import Gridap.TensorValues: double_contraction
import Gridap.TensorValues: symmetric_part

export Table
export PSpace
export UniformSampling
export NormalSampling
export realization
export ∂ₚt
export ∂ₚtt
export Nonaffine
export PTArray
export NonaffinePTArray
export AffinePTArray
export get_at_offsets
export recenter
export test_ptarray
export PTrialFESpace
export NonaffinePTrialFESpace
export AffinePTrialFESpace
export PMultiFieldFESpace
export field_offsets
export AbstractPTFunction
export PFunction
export PTFunction
export PTCellField
export GenericPTCellField
export PTOperationCellField
export PTFEFunction
export PTSingleFieldFEFunction
export PTTransientCellField
export PTTransientSingleFieldCellField
export PTSingleFieldTypes
export PTMultiFieldCellField
export PTMultiFieldFEFunction
export PTTransientMultiFieldCellField
export PTDomainContribution
export PTIntegrand
export ∫ₚ
export CollectionPTIntegrand
export PTTrialFESpace
export PTMultiFieldTrialFESpace
export PTFEOperator
export PTFEOperatorFromWeakForm
export AffinePTFEOperator
export NonlinearPTFEOperator
export get_residual
export get_jacobian
export linear_operator
export nonlinear_operator
export auxiliary_operator
export residual_for_trian!
export jacobian_for_trian!
export PODEOperator
export AffinePODEOperator
export PODEOpFromFEOp
export PTOperator
export get_ptoperator
export update_ptoperator
export _check_convergence
export PODESolver
export PThetaMethod
export PODESolution
export num_time_dofs
export get_times
export collect_residuals_for_trian
export collect_jacobians_for_trian
export PTThetaAffineMethodOperator
export PTThetaMethodOperator
export get_order
export get_L2_norm_matrix
export get_H1_norm_matrix
export ReducedMeasure

include("PSpace.jl")
include("PDiffOperators.jl")
include("PTArray.jl")
include("PFESpaces.jl")
include("PTFields.jl")
include("PTCellFields.jl")
include("PTDomainContribution.jl")
include("PTAssemblers.jl")
include("PTFESpaces.jl")
include("PTFEOperator.jl")
include("PODEOperatorInterface.jl")
include("PTOperator.jl")
include("PTSolvers.jl")
include("PAffineThetaMethod.jl")
include("PThetaMethod.jl")
include("FEUtils.jl")
include("ReducedMeasure.jl")
end # module
