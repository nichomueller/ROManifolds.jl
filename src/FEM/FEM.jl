module FEM

using DocStringExtensions

include("IndexMaps/IndexMaps.jl")

include("TProduct/TProduct.jl")

include("ParamDataStructures/ParamDataStructures.jl")

include("ParamAlgebra/ParamAlgebra.jl")

include("ParamFESpaces/ParamFESpaces.jl")

include("ParamODEs/ParamODEs.jl")

include("ParamUtils/ParamUtils.jl")

end # module
