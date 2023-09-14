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

import Base: +, -, *
import LinearAlgebra: fillstored!
import FillArrays:Fill,fill
import Statistics.mean
import GridapGmsh:GmshDiscreteModel
import Gridap.Helpers.@check
import Gridap.Helpers.@unreachable
import Gridap.Helpers.@notimplemented
import Gridap.Arrays:evaluate
import Gridap.Arrays:evaluate!
import Gridap.Algebra:InserterCSC
import Gridap.Algebra:LUNumericalSetup
import Gridap.Algebra:solve
import Gridap.Algebra:solve!
import Gridap.Algebra:residual!
import Gridap.Algebra:jacobian!
import Gridap.Algebra:allocate_jacobian
import Gridap.Algebra:allocate_residual
import Gridap.Algebra:allocate_vector
import Gridap.Algebra:allocate_matrix
import Gridap.CellData:ConstrainRowsMap
import Gridap.CellData:ConstrainColsMap
import Gridap.CellData:similar_cell_field
import Gridap.FESpaces:_pair_contribution_when_possible
import Gridap.FESpaces:assemble_vector
import Gridap.FESpaces:assemble_matrix
import Gridap.FESpaces:collect_cell_vector
import Gridap.FESpaces:collect_cell_matrix
import Gridap.FESpaces:get_fe_basis
import Gridap.FESpaces:get_trial_fe_basis
import Gridap.Geometry:GridView
import Gridap.MultiField:MultiFieldCellField
import Gridap.MultiField:MultiFieldFEBasisComponent
import Gridap.Polynomials:MonomialBasis
import Gridap.Polynomials:get_order
import Gridap.ODEs.TransientFETools:ODESolver
import Gridap.ODEs.TransientFETools:ODEOperator
import Gridap.ODEs.TransientFETools:OperatorType
import Gridap.ODEs.TransientFETools:TransientCellField
import Gridap.ODEs.TransientFETools:TransientSingleFieldCellField
import Gridap.ODEs.TransientFETools.TransientMultiFieldCellField
import Gridap.ODEs.TransientFETools:SingleFieldTypes,MultiFieldTypes
import Gridap.ODEs.TransientFETools:Affine
import Gridap.ODEs.TransientFETools:Nonlinear
import Gridap.ODEs.TransientFETools:solve_step!
import Gridap.ODEs.TransientFETools:allocate_trial_space
import Gridap.ODEs.TransientFETools:allocate_cache
import Gridap.ODEs.TransientFETools:fill_initial_jacobians
import Gridap.ODEs.TransientFETools:fill_jacobians
import Gridap.ODEs.TransientFETools:update_cache!
import Gridap.ODEs.TransientFETools:jacobians!
import Gridap.ODEs.ODETools:_allocate_matrix_and_vector
import Gridap.ODEs.TransientFETools._matdata_jacobian
import Gridap.ODEs.TransientFETools._vcat_matdata

include("FEOperations.jl")
include("PSpace.jl")
include("PDiffOperators.jl")
include("PTArray.jl")
include("PTCellFields.jl")
include("PTIntegration.jl")
include("PTAssemblers.jl")
include("PTrialFESpaces.jl")
include("PTFESpaces.jl")
include("PTFEOperator.jl")
include("PTFESolversInterface.jl")
include("PTSolvers.jl")
include("PODEQuantities.jl")
include("PAffineThetaMethod.jl")
include("PThetaMethod.jl")
include("Collectors.jl")
