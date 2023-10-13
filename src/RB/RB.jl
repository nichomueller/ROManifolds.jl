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

function single_field_rb_model(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver)

  # Offline phase
  printstyled("OFFLINE PHASE\n";bold=true,underline=true)
  if info.load_solutions
    sols,params = load(info,(Snapshots,Table))
  else
    nsnaps = info.nsnaps_state
    params = realization(feop,nsnaps)
    trial = get_trial(feop)
    sols,stats = collect_solutions(fesolver,feop,trial,params)
    save(info,(sols,params,stats))
  end
  if info.load_structures
    rbspace = load(info,RBSpace)
    rbrhs,rblhs = load(info,(RBAlgebraicContribution,Vector{RBAlgebraicContribution}))
  else
    rbspace = reduced_basis(info,feop,sols,params)
    rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
    save(info,(rbspace,rbrhs,rblhs))
  end

  # Online phase
  printstyled("ONLINE PHASE\n";bold=true,underline=true)
  test_rb_solver(info,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)
  return
end

function multi_field_rb_model(
  info::RBInfo,
  feop::PTFEOperator,
  fesolver::PODESolver)

  # Offline phase
  printstyled("OFFLINE PHASE\n";bold=true,underline=true)
  if info.load_solutions
    sols,params = load(info,(BlockSnapshots,Table))
  else
    nsnaps = info.nsnaps_state
    params = realization(feop,nsnaps)
    trial = get_trial(feop)
    sols,stats = collect_solutions(fesolver,feop,trial,params)
    save(info,(sols,params,stats))
  end
  if info.load_structures
    rbspace = load(info,RBSpace)
    rbrhs,rblhs = load(info,(VecBlockRBAlgebraicContribution,Vector{MatBlockRBAlgebraicContribution}))
  else
    rbspace = reduced_basis(info,feop,sols,params)
    rbrhs,rblhs = collect_compress_rhs_lhs(info,feop,fesolver,rbspace,sols,params)
    save(info,(rbspace,rbrhs,rblhs))
  end

  # Online phase
  printstyled("ONLINE PHASE\n";bold=true,underline=true)
  test_rb_solver(info,feop,fesolver,rbspace,rbrhs,rblhs,sols,params)
  return
end
