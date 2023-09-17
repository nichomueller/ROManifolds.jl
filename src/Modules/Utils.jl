module Utils
using LinearAlgebra
using SparseArrays
using Elemental
using Serialization

import Gridap.Helpers.@check

const Float = Float64
const EMatrix = Elemental.Matrix

export Float
export EMatrix

#Files
export get_parent_dir
export create_dir!
export save
export load
#Indexes
export from_vec_to_mat_idx
# Operations
export expand
# BasesConstruction
export orth_projection
export gram_schmidt
# NnzMatrix
export NnzMatrix
export compress
export compress!
export convert!
export get_nonzero_val
export recast
export tpod
export change_mode
# PStructure
export PStructure
export sum_contributions!

include("Files.jl")
include("Indexes.jl")
include("Operations.jl")
include("BasesConstruction.jl")
include("NnzMatrix.jl")
include("PStructure.jl")
end # module
