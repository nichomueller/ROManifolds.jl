module Extensions

using LinearAlgebra
using BlockArrays
using SparseArrays

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.Fields
using Gridap.FESpaces
using Gridap.Helpers
using Gridap.MultiField

using ROManifolds.DofMaps

export Extension
export ZeroExtension
export FunctionExtension
export HarmonicExtension
include("ExtensionInterface.jl")

export ExternalFESpace
export ExternalAgFEMSpace
include("ExternalFESpaces.jl")

export ExtensionFESpace
export ZeroExtensionFESpace
export FunctionExtensionFESpace
export HarmonicExtensionFESpace
export get_internal_space
export get_external_space
include("ExtensionFESpaces.jl")

export MultiFieldExtensionFESpace
include("MultiFieldExtensionFESpaces.jl")

export ExtensionAssembler
include("ExtensionAssemblers.jl")

export get_bg_dof_to_dof
export get_dof_to_bg_dof
include("DofUtils.jl")

include("ODofUtils.jl")

end # module
