module RB
using Mabla.FEM
using LinearAlgebra
using SparseArrays
using DrWatson
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
using Gridap.ODEs.ODETools
using Gridap.ODEs.TransientFETools

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

export AbstractTransientSnapshots
export BasicSnapshots
export TransientSnapshots
export TransientSnapshotsWithDirichletValues
export CompressedTransientSnapshots
export SelectedSnapshotsAtIndices
export InnerTimeOuterParamTransientSnapshots
export SelectedInnerTimeOuterParamTransientSnapshots
export NnzSnapshots
export GenericNnzSnapshots
export NnzSnapshotsSwappedColumns
export Snapshots
export slow_index
export fast_index
export tensor_getindex
export tensor_setindex!
export select_snapshots
export reverse_snapshots
export get_realization
export compress
export flatten
include("Snapshots.jl")

export RBInfo
export RBSolver
include("RBSolver.jl")

export RBSpace
export TestRBSpace
export TrialRBSpace
export reduced_fe_space
export reduced_basis
export get_basis_space
export num_reduced_space_dofs
export get_basis_time
export num_reduced_times
export recast
include("RBSpace.jl")

export RBOperator
export GalerkinProjectionOperator
export reduced_operator
export get_fe_trial
export get_fe_test
export allocate_fe_vector
export allocate_fe_matrix
export allocate_fe_matrix_and_vector
export fe_vector!
export fe_matrix!
export fe_matrices_and_vectors!
export fe_matrix_and_vector
include("GalerkinProjectionOperator.jl")

export ReducedIntegrationDomain
export AffineDecomposition
export AffineContribution
export affine_contribution
export mdeim
export get_mdeim_indices
export reduce_triangulation
export compress_basis_space
export combine_basis_time
export reduced_matrix_vector_form
export allocate_mdeim_coeff
export mdeim_coeff!
export allocate_mdeim_lincomb
export mdeim_lincomb!
include("AffineDecomposition.jl")

export ReducedOperator
include("ReducedOperator.jl")

export ComputationalStats
export RBResults
export rb_results
export load_solve
export generate_plots
include("PostProcess.jl")
end # module
