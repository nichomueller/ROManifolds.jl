module RB
using Mabla.FEM
using LinearAlgebra
using BlockArrays
using SparseArrays
using DrWatson
using Kronecker
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
using Gridap.ODEs

import Base: +,-,*,\
import StaticArrays: SVector
import UnPack: @unpack
import Gridap.Helpers: @abstractmethod,@check,@notimplemented,@unreachable
import Gridap.Arrays: evaluate!
import Gridap.Algebra: allocate_matrix,allocate_vector,solve
import PartitionedArrays: tuple_of_arrays

export tpod
export gram_schmidt!
export orth_complement!
export orth_projection
include("BasisConstruction.jl")

export compress_basis_space
export combine_basis_time
include("RBOperations.jl")

export AbstractSnapshots
export BasicSnapshots
export TransientSnapshots
export CompressedTransientSnapshots
export SelectedSnapshotsAtIndices
export NnzSnapshots
export GenericNnzSnapshots
export NnzSnapshotsSwappedColumns
export Snapshots
export tensor_getindex
export tensor_setindex!
export select_snapshots
export reverse_snapshots
export get_realization
export compress
include("Snapshots.jl")

export BDiagonal
include("BDiagonal.jl")

export TTSnapshots
export BasicTTSnapshots
export TransientTTSnapshots
include("TTSnapshots.jl")

export get_stage_operator
export jacobian_and_residual
include("ThetaMethod.jl")

export RBInfo
export RBSolver
export TTRBSolver
export get_test_directory
export fe_solutions
export ode_solutions
include("RBSolver.jl")

export Projection
export PODBasis
export TTSVDCores
export BlockProjection
export get_basis_space
export num_reduced_space_dofs
export get_basis_time
export num_reduced_times
include("Projection.jl")

export RBSpace
export reduced_fe_space
export reduced_basis
export recast
include("RBSpace.jl")

export RBOperator
export PODOperator
export reduced_operator
export get_fe_trial
export get_fe_test
export allocate_fe_residual
export allocate_fe_jacobian
export allocate_fe_jacobian_and_residual
export fe_residual
export fe_residual!
export fe_jacobian
export fe_jacobian!
export fe_jacobian!
export fe_jacobian_and_residual!
export fe_jacobian_and_residual
include("PODOperator.jl")

export ReducedIntegrationDomain
export AffineDecomposition
export AffineContribution
export BlockAffineDecomposition
export mdeim
export get_mdeim_indices
export reduce_triangulation
export reduced_matrix_vector_form
export allocate_mdeim_coeff
export mdeim_coeff!
export allocate_mdeim_lincomb
export mdeim_lincomb!
include("AffineDecomposition.jl")

export PODMDEIMOperator
export LinearNonlinearPODMDEIMOperator
include("PODMDEIMOperator.jl")

export RBNonlinearOperator
export RBThetaMethodParamOperator
include("RBNonlinearOperator.jl")

export ComputationalStats
export RBResults
export rb_results
export load_solve
export generate_plots
include("PostProcess.jl")
end # module
