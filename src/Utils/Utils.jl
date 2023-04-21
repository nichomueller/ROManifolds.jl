using DataFrames
using FillArrays
using LinearAlgebra
using Distributions
using Random
using SuiteSparse
using SparseArrays
using Arpack
using Elemental
using DelimitedFiles
using PartitionedArrays,SharedArrays
using Test
using PlotlyJS
using ForwardDiff
using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.Io
using GridapGmsh
using GridapDistributed,GridapPETSc
using GridapP4est
using Gridap.TensorValues
using Gridap.ODEs.TransientFETools

import Base.Threads.@threads
import Gridap:solve!
import Gridap:∇
import Gridap.Algebra:allocate_matrix
import Gridap.Algebra:allocate_vector
import Gridap.Algebra:AffineOperator
import Gridap.Algebra:NonlinearOperator
import Gridap.Algebra:LinearSolver
import Gridap.Algebra:NonlinearSolver
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
import LineSearches:BackTracking

const Float = Float64
const EMatrix = Elemental.DistMatrix

include("Files.jl")
include("Indexes.jl")
include("Operations.jl")
include("BasesConstruction.jl")
include("SystemSolvers.jl")
include("Plots.jl")
