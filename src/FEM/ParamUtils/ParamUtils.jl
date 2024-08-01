module ParamUtils

using LinearAlgebra

using Gridap
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Helpers

using Mabla.FEM.TProduct
using Mabla.FEM.ParamDataStructures

import BlockArrays: BlockArray, Block, blocksize

include("LagrangianDofBases.jl")

include("TriangulationView.jl")

export PartialFunctions
export PartialDerivative
export PartialTrace
export ∂ₓ₁, ∂ₓ₂, ∂ₓ₃
include("Operations.jl")

end # module
