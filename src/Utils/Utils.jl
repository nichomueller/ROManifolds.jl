using DataFrames
using FillArrays
using LinearAlgebra
using Distributions
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
using LineSearches:BackTracking
import Gridap:∇

const Float = Float64

include("Files.jl")
include("Indexes.jl")
include("Operations.jl")
include("Plots.jl")
