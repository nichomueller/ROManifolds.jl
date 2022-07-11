#= module Utils

using CSV
using Distributions
using LinearAlgebra
using Plots
using SparseArrays
using TensorOperations
using Gridap
using Gridap.Algebra

export create_dir
export get_subdirectories
export load_CSV
export save_CSV
export append_CSV
export from_vec_to_mat_idx
export from_full_idx_to_sparse_idx
export remove_zero_entries
export label_sorted_elems
export mydot
export mynorm
export matrix_product
export generate_Parameter
export plot_R²_R
export plot_R_R²
export generate_and_save_plot

include("files.jl")
include("indexes.jl")
include("operations.jl")
include("plots.jl")

end =#

using DataFrames
using FillArrays
using LinearAlgebra
using Distributions
using SuiteSparse
using SparseArrays
using Arpack
using CSV
using JLD
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

include("files.jl")
include("indexes.jl")
include("operations.jl")
include("plots.jl")
