module FEM

using DocStringExtensions

include("ParamDataStructures/ParamDataStructures.jl")

include("ParamAlgebra/ParamAlgebra.jl")

include("ParamFESpaces/ParamFESpaces.jl")

include("ParamODEs/ParamODEs.jl")

# include("ParamTensorProduct/ParamTensorProduct.jl")

include("ParamUtils/ParamUtils.jl")

end # module
