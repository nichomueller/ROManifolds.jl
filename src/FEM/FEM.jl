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
using Gridap.ODEs.TransientFETools

const Float = Float64

import Base: inv,abs,abs2,*,+,-,/,adjoint,transpose,real,imag,conj
import LinearAlgebra: det,tr,cross,dot,⋅,rmul!,fillstored!
import Distributions: Uniform,Normal
import ForwardDiff : derivative
import FillArrays: Fill,fill
import BlockArrays: blockedrange
import Statistics: mean
import UnPack: @unpack
import GridapGmsh: GmshDiscreteModel
import Gridap.Helpers: @abstractmethod,@check,@notimplemented,@unreachable
import Gridap.Arrays: CachedArray,lazy_map,testitem,return_value,return_cache,evaluate!,evaluate,getindex!,setsize!
import Gridap.Algebra: InserterCSC,LUNumericalSetup,solve,solve!,residual!,jacobian!,allocate_jacobian,allocate_residual,allocate_vector,allocate_matrix,numerical_setup,numerical_setup!
import Gridap.CellData: ConstrainRowsMap,ConstrainColsMap,OperationCellField,GenericMeasure,similar_cell_field,_get_cell_points,_operate_cellfields,_to_common_domain
import Gridap.FESpaces: _pair_contribution_when_possible,assemble_vector,assemble_matrix,collect_cell_vector,collect_cell_matrix,get_fe_basis,get_trial_fe_basis
import Gridap.Geometry: FaceToCellGlue,TriangulationView,GridView,DiscreteModelPortion,_compute_face_to_q_vertex_coords
import Gridap.MultiField: MultiFieldCellField,MultiFieldFEBasisComponent,num_fields
import Gridap.Polynomials: get_order
import Gridap.ODEs.ODETools: _allocate_matrix_and_vector,∂t,∂tt
import Gridap.ODEs.TransientFETools: ODESolver,ODEOperator,OperatorType,TransientCellField,TransientSingleFieldCellField,TransientMultiFieldCellField
import Gridap.ODEs.TransientFETools: SingleFieldTypes,MultiFieldTypes,Affine,Nonlinear,solve_step!,allocate_trial_space,allocate_cache
import Gridap.ODEs.TransientFETools: fill_jacobians,update_cache!,jacobians!,_matdata_jacobian,_vcat_matdata,_to_transient_single_fields
import Gridap.TensorValues: inner,outer,double_contraction,symmetric_part

export PSpace,UniformSampling,NormalSampling,realization
export ∂ₚt,∂ₚtt
export Nonaffine,PTArray,NonaffinePTArray,AffinePTArray,get_at_offsets,recenter,test_ptarray
export PTrialFESpace,NonaffinePTrialFESpace,AffinePTrialFESpace,PMultiFieldFESpace
export AbstractPTFunction,PFunction,PTFunction
export PTCellField,GenericPTCellField,PTOperationCellField,PTFEFunction,PTSingleFieldFEFunction,PTTransientCellField,PTTransientSingleFieldCellField,PTSingleFieldTypes

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
include("PODEQuantities.jl")
include("PAffineThetaMethod.jl")
include("PThetaMethod.jl")
include("FEOperations.jl")
include("ReducedMeasure.jl")
end # module
