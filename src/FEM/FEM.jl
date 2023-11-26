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
import Gridap.Arrays: CachedArray,lazy_map,testitem,return_value,return_cache,evaluate!,evaluate,getindex!,setsize!,get_array,create_from_nz
import Gridap.Algebra: InserterCSC,LUNumericalSetup,solve,solve!,residual!,jacobian!,allocate_jacobian,allocate_residual,allocate_vector,allocate_matrix,numerical_setup,numerical_setup!
import Grida.Fields: integrate
import Gridap.CellData: CellField,GenericMeasure,CompositeMeasure,ConstrainRowsMap,ConstrainColsMap,DomainStyle,OperationCellField,gradient,∇∇,add_contribution!,get_contribution,move_contributions,get_data,get_domains,num_domains,change_domains,similar_cell_field,_get_cell_points,_operate_cellfields,_to_common_domain
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction,assemble_vector,assemble_matrix,numeric_loop_vector!,numeric_loop_matrix!,collect_cell_vector,collect_cell_matrix,get_cell_dof_values,get_fe_basis,get_trial_fe_basis,get_triangulation,_pair_contribution_when_possible,scatter_free_and_dirichlet_values,gather_dirichlet_values!,zero_free_values,zero_dirichlet_values,compute_dirichlet_values_for_tags!,num_free_dofs,get_free_dof_ids,get_cell_dof_ids,get_cell_isconstrained,get_cell_constraints,get_cell_is_dirichlet,interpolate_everywhere,interpolate_dirichlet,interpolate!
import Gridap.Geometry: FaceToCellGlue,TriangulationView,GridView,DiscreteModelPortion,_compute_face_to_q_vertex_coords
import Gridap.MultiField: MultiFieldCellField,MultiFieldFEBasisComponent,MultiFieldStyle,num_fields,restrict_to_field,compute_field_offsets
import Gridap.ReferenceFEs: get_order
import Gridap.ODEs.ODETools: _allocate_matrix_and_vector,∂t,∂tt
import Gridap.ODEs.TransientFETools: ODESolver,ODEOperator,OperatorType,TransientCellField,TransientSingleFieldCellField,TransientMultiFieldCellField,TransientFEBasis
import Gridap.ODEs.TransientFETools: SingleFieldTypes,MultiFieldTypes,Affine,Nonlinear,solve_step!,allocate_trial_space,allocate_cache
import Gridap.ODEs.TransientFETools: fill_jacobians,fill_initial_jacobians,update_cache!,jacobians!,_matdata_jacobian,_vcat_matdata,_to_transient_single_fields
import Gridap.TensorValues: inner,outer,double_contraction,symmetric_part

export PSpace,UniformSampling,NormalSampling,realization
export ∂ₚt,∂ₚtt
export Nonaffine,PTArray,NonaffinePTArray,AffinePTArray,get_at_offsets,recenter,test_ptarray
export PTrialFESpace,NonaffinePTrialFESpace,AffinePTrialFESpace,PMultiFieldFESpace
export AbstractPTFunction,PFunction,PTFunction
export PTCellField,GenericPTCellField,PTOperationCellField,PTFEFunction,PTSingleFieldFEFunction,PTTransientCellField,PTTransientSingleFieldCellField,PTSingleFieldTypes,PTMultiFieldCellField,PTMultiFieldFEFunction,PTTransientMultiFieldCellField
export PTDomainContribution,PTIntegrand,∫ₚ,CollectionPTIntegrand
export PTTrialFESpace,PTMultiFieldTrialFESpace
export PTFEOperator,PTFEOperatorFromWeakForm,AffinePTFEOperator,NonlinearPTFEOperator,get_residual,get_jacobian,linear_operator,nonlinear_operator,auxiliary_operator,residual_for_trian!,jacobian_for_trian!
export PODEOperator,AffinePODEOperator,PODEOpFromFEOp
export PTOperator,get_ptoperator,update_ptoperator
export _check_convergence
export PODESolver,PThetaMethod,PODESolution,num_time_dofs,get_times,collect_single_field_solutions,collect_multi_field_solutions,collect_residuals_for_trian,collect_jacobians_for_trian
export PTThetaAffineMethodOperator,PTThetaMethodOperator
export get_discrete_model,set_labels!,field_offsets,get_order,get_L2_norm_matrix,get_H1_norm_matrix
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
include("PODEQuantities.jl")
include("PAffineThetaMethod.jl")
include("PThetaMethod.jl")
include("FEOperations.jl")
include("ReducedMeasure.jl")
end # module
