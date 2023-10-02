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

function reduced_basis_model(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver)

  # Offline phase
  if info.load_structures
    sols,params,rbspace,rbrhs,rblhs = load(info,(
      AbstractSnapshots,
      Table,
      AbstractRBSpace,
      AbstractRBAlgebraicContribution,
      AbstractRBAlgebraicContribution))
  end
  nsnaps = info.nsnaps_state
  params = realization(feop,nsnaps)
  sols = collect_solutions(fesolver,feop,params)
  rbspace = get_reduced_basis(info,feop,sols,fesolver,params)
  rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
  save(info,(sols,params,rbspace,rbrhs,rblhs))

  # Online phase
  test_rb_solver(info,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)
  return
end
