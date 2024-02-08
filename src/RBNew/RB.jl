module RB
using Mabla.FEM
using LinearAlgebra
using SparseArrays
using DrWatson
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

export RBInfo
export RBSolver
include("RBSolver.jl")

export tpod
export gram_schmidt!
export orth_complement!
export orth_projection
include("BasisConstruction.jl")

export AbstractTransientSnapshots
export TransientSnapshotsWithInitialValues
export TransientSnapshots
export TransientSnapshotsWithDirichletValues
export CompressedTransientSnapshots
export NnzTransientSnapshots
export Snapshots
include("Snapshots.jl")

export RBSpace
export TestRBSpace
export TrialRBSpace
export reduced_fe_space
export reduced_basis
include("RBSpace.jl")

export RBOperator
export GalerkinProjectionOperator
export reduced_operator
export collect_matrices_vectors!
include("GalerkinProjectionOperator.jl")

export AffineDecomposition
include("AffineDecomposition.jl")

export ReducedOperator
include("ReducedOperator.jl")
end # module
