module FEM

using DocStringExtensions

include("Utils/Utils.jl")

include("IndexMaps/IndexMaps.jl")

include("TProduct/TProduct.jl")

include("ParamDataStructures/ParamDataStructures.jl")

include("ParamAlgebra/ParamAlgebra.jl")

include("ParamFESpaces/ParamFESpaces.jl")

include("ParamSteady/ParamSteady.jl")

include("ParamODEs/ParamODEs.jl")

end # module
