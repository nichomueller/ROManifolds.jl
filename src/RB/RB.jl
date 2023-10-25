using LinearAlgebra
using SparseArrays
using Serialization
using NearestNeighbors
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField

import StaticArrays: SVector
import Gridap.Helpers:@check,@unreachable
import Gridap.Arrays:Table,evaluate!
import Gridap.Algebra:allocate_matrix,allocate_vector,solve
import Gridap.ODEs.TransientFETools:Affine,TransientFETools,ODESolver

include("RBInfo.jl")
include("Snapshots.jl")
include("NnzArrays.jl")
include("RBSpaces.jl")
include("RBAffineDecomposition.jl")
include("RBAlgebraicContribution.jl")
include("RBResults.jl")
include("RBBlocks.jl")
