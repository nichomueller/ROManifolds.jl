module RB
using Mabla.Utils
using Mabla.FEM

using LinearAlgebra
using SparseArrays
using Elemental
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

import Gridap.Helpers.@check
import Gridap.Helpers.@unreachable
import Gridap.Arrays:Table
import Gridap.Arrays:evaluate!
import Gridap.Algebra:allocate_jacobian
import Gridap.Algebra:allocate_residual
import Gridap.Algebra:solve
import Gridap.ODEs.TransientFETools:Affine
import Gridap.ODEs.TransientFETools:ODESolver

import Mabla.Utils:tpod
import Mabla.Utils:compress
import Mabla.Utils:recast

# Collectors
export collect_solutions
export collect_residuals
export collect_jacobians
# RBInfo
export RBInfo
export load_test
export save_test
# RBSpaces
export compress_solutions
# RBAffineDecomposition
export compress_residuals
export compress_jacobians
export compress_component
# RBOperators
export TransientRBOperator
export reduce_fe_operator
# RBResults
export test_rb_operator

include("Snapshots.jl")
include("Collectors.jl")
include("RBInfo.jl")
include("RBSpaces.jl")
include("RBAffineDecomposition.jl")
include("RBAlgebraicContributions.jl")
include("RBOperators.jl")
include("RBResults.jl")
end # module
