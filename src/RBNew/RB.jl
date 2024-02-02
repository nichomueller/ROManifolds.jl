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
import Gridap.Helpers: @check,@notimplemented,@unreachable
import Gridap.Arrays: evaluate!
import Gridap.Algebra: allocate_matrix,allocate_vector,solve
import PartitionedArrays: tuple_of_arrays

export RBInfo
export get_parent_dir
export create_dir
export correct_path
include("RBInfo.jl")

export tpod
export gram_schmidt!
export orth_complement!
export orth_projection
include("BasisConstruction.jl")

export AbstractTransientSnapshots
export TransientSnapshotsWithInitialValues
export TransientSnapshots
export Snapshots
include("Snapshots.jl")

export reduced_basis
export RBSpace
export SingleFieldRBSpace
include("RBSpace.jl")
end # module
