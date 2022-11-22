using DataFrames
using FillArrays
using LinearAlgebra
using Distributions
using Random
using SuiteSparse
using SparseArrays
using Arpack
using DelimitedFiles
using Parameters
using Test
using ScatteredInterpolation
using PlotlyJS
using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.Io
using GridapGmsh
using Gridap.TensorValues
using Gridap.ODEs.TransientFETools

import Gridap:∇
import Gridap.ODEs.TransientFETools:TransientCellField
import Gridap.Algebra:LinearSolver
import Gridap.Algebra:NonlinearSolver
import LineSearches:BackTracking
import Gridap.ODEs.TransientFETools:ODEOperator
import Gridap.ODEs.TransientFETools:evaluate!

#=
import Gridap.ODEs.TransientFETools:get_algebraic_operator
import Gridap.ODEs.TransientFETools:SparseMatrixAssembler
import Gridap.ODEs.TransientFETools:evaluate
import Gridap.ODEs.TransientFETools:evaluate!
import Gridap.ODEs.TransientFETools:allocate_jacobian
import Gridap.ODEs.TransientFETools:allocate_residual
import Gridap.ODEs.TransientFETools:jacobian!
import Gridap.ODEs.TransientFETools:jacobians!
import Gridap.ODEs.TransientFETools:fill_initial_jacobians
import Gridap.ODEs.TransientFETools:fill_jacobians
import Gridap.ODEs.TransientFETools:residual! =#

const Float = Float64

include("Files.jl")
include("Indexes.jl")
include("Operations.jl")
include("Plots.jl")
