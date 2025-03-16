module Extensions

using LinearAlgebra
using BlockArrays
using SparseArrays

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.FESpaces
using Gridap.Helpers
using Gridap.MultiField

using ROManifolds.ParamDataStructures

export ExtensionFESpace
export ZeroExtensionFESpace
export FunctionExtensionFESpace
export HarmonicExtensionFESpace
include("ExtensionFESpaces.jl")

export MultiFieldExtensionFESpace
include("MultiFieldExtensionFESpaces.jl")

include("ExtensionSolvers.jl")

end # module
