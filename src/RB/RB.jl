using LinearAlgebra
using SparseArrays
using Serialization
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField

import BlockArrays:BlockVector,BlockMatrix,BlockArray,mortar
import Gridap.Helpers.@check
import Gridap.Helpers.@unreachable
import Gridap.Arrays:Table
import Gridap.Arrays:evaluate!
import Gridap.Algebra:allocate_jacobian
import Gridap.Algebra:allocate_residual
import Gridap.Algebra:solve
import Gridap.ODEs.TransientFETools:Affine
import Gridap.ODEs.TransientFETools:ODESolver

include("Snapshots.jl")
include("NnzArrays.jl")
include("RBInfo.jl")
include("RBSpaces.jl")
include("RBAffineDecomposition.jl")
include("RBAlgebraicContribution.jl")
include("RBOperators.jl")
# include("RBResults.jl")
