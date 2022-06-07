module Mabla

using Logging
using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FEMSpaces
using Gridap.Fields
using Gridap.Geometry
using GridapGmsh
using Gridap.Helpers
using Gridap.Io
using Gridap.ReferenceFEs
using Gridap.TensorValues
import Gridap:∇

include("Utils/Utils.jl")
include("FEM/FEM.jl")
include("RB/RB.jl")

export FEM_paths

end
