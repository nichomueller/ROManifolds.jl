module Utils

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

end
