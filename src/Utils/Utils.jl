using LinearAlgebra
using SparseArrays
using Elemental
using Serialization

import Gridap.Helpers.@check

const Float = Float64
const EMatrix = Elemental.Matrix

include("Files.jl")
include("Indexes.jl")
include("Operations.jl")
include("BasesConstruction.jl")
include("NnzArray.jl")
