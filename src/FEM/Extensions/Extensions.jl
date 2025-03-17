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

using ROManifolds.Utils
using ROManifolds.DofMaps
using ROManifolds.TProduct
using ROManifolds.ParamDataStructures
using ROManifolds.ParamAlgebra
using ROManifolds.ParamFESpaces
using ROManifolds.ParamSteady

export Extension
export GenericExtension
export ZeroExtension
export FunctionExtension
export HarmonicExtension
include("ExtensionInterface.jl")

export ParamExtension
export UnEvalExtension
include("ParamExtensions.jl")

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

# export ExtensionAssembler
# include("ExtensionAssemblers.jl")

export get_bg_dof_to_dof
export get_dof_to_bg_dof
include("DofUtils.jl")

include("ODofUtils.jl")

end # module
