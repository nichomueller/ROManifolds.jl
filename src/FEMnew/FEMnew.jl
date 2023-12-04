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
import Gridap.Fields: BroadcastingFieldOpMap
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

include("PSpace.jl")
include("PDiffOperators.jl")
include("PTArray.jl")
include("PTFields.jl")
include("PTCellFields.jl")
