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
include("RBBlocks.jl")
include("RBResults.jl")

function reduced_basis_model(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver)

  # Offline phase
  if info.load_solutions
    sols,params = load(info,(Snapshots,Table))
  else
    nsnaps = info.nsnaps_state
    params = realization(feop,nsnaps)
    test = get_test(feop)
    sols,stats = collect_solutions(fesolver,feop,test,params)
    save(info,(sols,params,stats))
  end
  if info.load_structures
    rbspace = load(info,RBSpace)
    rbrhs,rblhs = load(info,(RBAlgebraicContribution,Vector{RBAlgebraicContribution}))
  else
    rbspace = reduced_basis(info,feop,sols,fesolver,params)
    rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
    save(info,(rbspace,rbrhs,rblhs))
  end

  # Online phase
  test_rb_solver(info,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)
  return
end
