using Revise
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
import Gridap:solve!
import Gridap:∇
import Gridap.Arrays:Table
import Gridap.Algebra:allocate_matrix
import Gridap.Algebra:allocate_vector
import Gridap.Algebra:AffineOperator
import Gridap.Algebra:NonlinearOperator
import Gridap.Algebra:LinearSolver
import Gridap.Algebra:NonlinearSolver
import Gridap.Algebra:InserterCSC
import Gridap.Algebra:create_from_nz
import Gridap.Algebra:nz_allocation
import Gridap.Algebra:nz_counter
import Gridap.FESpaces:_pair_contribution_when_possible
import Gridap.FESpaces.length_to_ptrs!
import Gridap.Polynomials:MonomialBasis
import Gridap.ODEs.TransientFETools:ODESolver
import Gridap.ODEs.TransientFETools:OperatorType
import Gridap.ODEs.TransientFETools:Affine
import Gridap.ODEs.TransientFETools:Nonlinear
import Gridap.ODEs.TransientFETools:solve_step!
import Gridap.ODEs.TransientFETools:ODEOperator
import Gridap.ODEs.TransientFETools:evaluate!
import Gridap.ODEs.TransientFETools:allocate_trial_space
import Gridap.ODEs.TransientFETools:allocate_cache
import Gridap.ODEs.TransientFETools:allocate_jacobian
import Gridap.ODEs.TransientFETools:allocate_residual
import Gridap.ODEs.TransientFETools:fill_initial_jacobians
import Gridap.ODEs.TransientFETools:fill_jacobians
import Gridap.ODEs.TransientFETools:get_order
import Gridap.ODEs.TransientFETools:TransientCellField
import Gridap.ODEs.TransientFETools:update_cache!
import Gridap.ODEs.TransientFETools:residual!
import Gridap.ODEs.TransientFETools:jacobian!
import Gridap.ODEs.TransientFETools:jacobians!
import Gridap.ODEs.TransientFETools._vcat_matdata
import Gridap.ODEs.TransientFETools._matdata_jacobian
import LineSearches:BackTracking

const Float = Float64
const EMatrix = Elemental.Matrix

include("Files.jl")
include("Indexes.jl")
include("Operations.jl")
include("SystemSolvers.jl")
include("NnzArray.jl")
include("Snapshots.jl")
