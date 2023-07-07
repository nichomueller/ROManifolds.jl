using DataFrames
using FillArrays
using LinearAlgebra
using Distributions
using Random
using SuiteSparse
using SparseArrays
using Elemental
using DelimitedFiles
using Serialization
using ForwardDiff
using BlockArrays
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

import Base.Threads.@threads
import Gridap.Helpers.@check
import Gridap.Helpers.@notimplemented
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
import Gridap.FESpaces:get_fe_basis
import Gridap.FESpaces:get_trial_fe_basis
import Gridap.Geometry:GridView
import Gridap.MultiField:MultiFieldFEBasisComponent
import Gridap.Polynomials:MonomialBasis
import Gridap.Polynomials:get_order
import Gridap.ODEs.TransientFETools:ODESolver
import Gridap.ODEs.TransientFETools:ODEOperator
import Gridap.ODEs.TransientFETools:OperatorType
import Gridap.ODEs.TransientFETools:TransientCellField
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
import Gridap.ODEs.TransientFETools:∂t
import Gridap.ODEs.TransientFETools:∂tt
import LineSearches:BackTracking

const Float = Float64
const EMatrix = Elemental.Matrix

include("Files.jl")
include("Indexes.jl")
include("Operations.jl")
include("BasesConstruction.jl")
include("SystemSolvers.jl")
include("NnzArray.jl")
include("ParamArray.jl")
include("Affinity.jl")
