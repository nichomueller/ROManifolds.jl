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
import Arrays:evaluate
import Arrays:evaluate!
import Algebra:InserterCSC
import Algebra:LUNumericalSetup
import Algebra:solve
import Algebra:solve!
import Algebra:residual!
import Algebra:jacobian!
import Algebra:allocate_jacobian
import Algebra:allocate_residual
import Algebra:allocate_vector
import Algebra:allocate_matrix
import FESpaces:_pair_contribution_when_possible
import FESpaces:assemble_vector
import FESpaces:assemble_matrix
import FESpaces:collect_cell_vector
import FESpaces:collect_cell_matrix
import FESpaces:get_fe_basis
import FESpaces:get_trial_fe_basis
import Geometry:GridView
import MultiField:MultiFieldCellField
import MultiField:MultiFieldFEBasisComponent
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
# PSpace
export SamplingStyle
export UniformSampling
export realization
# DiffOperators
export time_derivative
# PTFESpaces
export PTTrialFESpace
# PTFEOperator
export PTFEOperator
# FilteredPTFEOperator
export FilteredPTFEOperator
export allocate_evaluation_function
export evaluation_function
export filter_evaluation_function
export collect_trian_res
export collect_trian_jac
# TimeMarchingSchemes
export ThetaMethod
export get_time_ndofs
export get_times
# FECollectors
export CollectSolutionMap
# Affinity
export Affinity
export ZeroAffinity
export PAffinity
export TimeAffinity
export PTimeAffinity
export NonAffinity
export affinity_residual
export affinity_jacobian
# LagrangianQuadFESpaces
export LagrangianQuadRefFE
export get_phys_quad_points

include("FEOperations.jl")
include("PSpace.jl")
include("DiffOperators.jl")
include("PTFESpaces.jl")
include("PTFEOperator.jl")
include("PTFESolversInterface.jl")
include("PTFESolvers.jl")
include("FilteredPTFEOperator.jl")
include("FECollectors.jl")
include("TimeMarchingSchemes.jl")
include("NothingIntegration.jl")
include("AffineThetaMethod.jl")
include("ThetaMethod.jl")
include("Affinity.jl")
include("LagrangianQuadFESpaces.jl")
end # module
