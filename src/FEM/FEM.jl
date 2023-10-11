using LinearAlgebra
using Distributions
using ForwardDiff
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.Io
using GridapDistributed
using Gridap.ODEs.TransientFETools

const Float = Float64

import Base: inv,abs,abs2,*,+,-,/,adjoint,transpose,real,imag,conj
import LinearAlgebra: det,tr,cross,dot,⋅,rmul!,fillstored!
import FillArrays: Fill,fill
import BlockArrays: blockedrange
import Statistics: mean
import GridapGmsh: GmshDiscreteModel
import Gridap.Helpers: @abstractmethod,@check,@notimplemented,@unreachable
import Gridap.Arrays: evaluate,evaluate!
import Gridap.Algebra: InserterCSC,LUNumericalSetup,solve,solve!,residual!,jacobian!,allocate_jacobian,allocate_residual,allocate_vector,allocate_matrix
import Gridap.CellData: ConstrainRowsMap,ConstrainColsMap,OperationCellField,GenericMeasure,similar_cell_field,_get_cell_points,_operate_cellfields,_to_common_domain
import Gridap.FESpaces: _pair_contribution_when_possible,assemble_vector,assemble_matrix,collect_cell_vector,collect_cell_matrix,get_fe_basis,get_trial_fe_basis
import Gridap.MultiField: MultiFieldCellField,MultiFieldFEBasisComponent
import Gridap.Polynomials: get_order
import Gridap.ODEs.ODETools: _allocate_matrix_and_vector
import Gridap.ODEs.TransientFETools: ODESolver,ODEOperator,OperatorType,TransientCellField,TransientSingleFieldCellField,TransientMultiFieldCellField
import Gridap.ODEs.TransientFETools: SingleFieldTypes,MultiFieldTypes,Affine,Nonlinear,solve_step!,allocate_trial_space,allocate_cache
import Gridap.ODEs.TransientFETools: fill_initial_jacobians,fill_jacobians,update_cache!,jacobians!,_matdata_jacobian,_vcat_matdata,_to_transient_single_fields
import Gridap.TensorValues: inner,outer,double_contraction,symmetric_part

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
include("PTSolvers.jl")
include("PODEQuantities.jl")
include("PAffineThetaMethod.jl")
include("PThetaMethod.jl")
include("FEOperations.jl")
