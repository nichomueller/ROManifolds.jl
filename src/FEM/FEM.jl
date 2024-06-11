module FEM

using DocStringExtensions

include("ParamDataStructures/ParamDataStructures.jl")

include("ParamAlgebra/ParamAlgebra.jl")

include("IndexMaps/IndexMaps.jl")

include("ParamTProduct/ParamTProduct.jl")

include("ParamFESpaces/ParamFESpaces.jl")

include("ParamODEs/ParamODEs.jl")

include("ParamUtils/ParamUtils.jl")

end # module
