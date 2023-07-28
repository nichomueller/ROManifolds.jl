module FEM
using Mabla.Utils

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
using GridapGmsh
using GridapDistributed
using GridapPETSc
using GridapP4est
using Gridap.TensorValues
using Gridap.ODEs.TransientFETools

import Gridap.Helpers.@check
import Gridap.Helpers.@unreachable
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
import Gridap.ODEs.TransientFETools.TransientMultiFieldCellField
import Gridap.ODEs.TransientFETools:Affine
import Gridap.ODEs.TransientFETools:Nonlinear
import Gridap.ODEs.TransientFETools:solve_step!
import Gridap.ODEs.TransientFETools:allocate_trial_space
import Gridap.ODEs.TransientFETools:allocate_cache
import Gridap.ODEs.TransientFETools:fill_initial_jacobians
import Gridap.ODEs.TransientFETools:fill_jacobians
import Gridap.ODEs.TransientFETools:update_cache!
import Gridap.ODEs.TransientFETools:jacobians!
import Gridap.ODEs.TransientFETools._matdata_jacobian
import Gridap.ODEs.TransientFETools._vcat_matdata
import Gridap.ODEs.TransientFETools:∂t
import Gridap.ODEs.TransientFETools:∂tt

# FEOperations
export collect_cell_contribution
export collect_trian
export is_parent
export modify_measures
export is_change_possible
export get_discrete_model
# ParamSpace
export SamplingStyle
export UniformSampling
export realization
# DiffOperators
export time_derivative
# ParamTransientFESpaces
export ParamTransientTrialFESpace
# ParamTransientFEOperator
export ParamTransientFEOperator
# FilteredParamTransientFEOperator
export FilteredParamTransientFEOperator
export allocate_evaluation_function
export evaluation_function
export filter_evaluation_function
export collect_trian_res
export collect_trian_jac
# TimeMarchingSchemes
export θMethod
export get_time_ndofs
export get_times
# FECollectors
export CollectSolutionMap
# Affinity
export Affinity
export ZeroAffinity
export ParamAffinity
export TimeAffinity
export ParamTimeAffinity
export NonAffinity
export affinity_residual
export affinity_jacobian
# LagrangianQuadFESpaces
export LagrangianQuadRefFE
export get_phys_quad_points

include("FEOperations.jl")
include("ParamSpace.jl")
include("DiffOperators.jl")
include("ParamTransientFESpaces.jl")
include("ParamTransientFEOperator.jl")
include("ParamTransientFESolversInterface.jl")
include("ParamTransientFESolvers.jl")
include("FilteredParamTransientFEOperator.jl")
include("FECollectors.jl")
include("TimeMarchingSchemes.jl")
include("NothingIntegration.jl")
include("AffineThetaMethod.jl")
include("ThetaMethod.jl")
include("Affinity.jl")
include("LagrangianQuadFESpaces.jl")
end # module
