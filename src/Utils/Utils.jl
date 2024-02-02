module Utils
using LinearAlgebra
using SparseArrays
using Serialization

import Gridap.Helpers.@check
import Gridap.Arrays.LazyArray

const Float64 = Float64

export Float64
export get_parent_dir
export create_dir
export correct_path
export save
export load
export num_active_dirs
export vec_to_mat_idx
export slow_idx
export fast_idx
export index_pairs
export change_mode
export compress_array
export recenter
export tpod
export gram_schmidt!
export orth_complement!
export orth_projection

include("Files.jl")
include("Indexes.jl")
include("Operations.jl")
include("BasesConstruction.jl")
end # module
