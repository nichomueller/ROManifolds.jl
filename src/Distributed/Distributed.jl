module Distributed
using Mabla.FEM
using Mabla.RB

using LinearAlgebra
using SparseArrays
using Serialization
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
using GridapDistributed
using PartitionedArrays

import Base: +
import Base: -
import LinearAlgebra: fillstored!
import SparseArrays: AbstractSparseMatrixCSC
import SparseMatricesCSR: SparseMatrixCSR
import SparseMatricesCSR: getoffset
import UnPack: @unpack

import Gridap.Helpers: @check,@notimplemented,@unreachable

import Mabla.FEM: ParamBroadcast

import PartitionedArrays: SubSparseMatrix
import PartitionedArrays: EltypeVector
import PartitionedArrays: NZIteratorCSC
import PartitionedArrays: NZIteratorCSR
import PartitionedArrays: VectorAssemblyCache
import PartitionedArrays: SparseMatrixAssemblyCache
import PartitionedArrays: PBroadcasted
import PartitionedArrays: is_consistent
import PartitionedArrays: allocate_exchange_impl
import PartitionedArrays: exchange_impl!
import PartitionedArrays: exchange_fetch
import PartitionedArrays: assemble_coo!
import PartitionedArrays: allocate_local_values
import PartitionedArrays: assembly_buffers
import PartitionedArrays: getany
import PartitionedArrays: trivial_partition
import PartitionedArrays: to_trivial_partition
import PartitionedArrays: length_to_ptrs!
import PartitionedArrays: rewind_ptrs!

import GridapDistributed: BlockPVector
import GridapDistributed: DistributedDiscreteModel
import GridapDistributed: LocalView
import GridapDistributed: PSparseMatrixBuilderCOO
import GridapDistributed: DistributedCounterCOO
import GridapDistributed: PVectorBuilder
import GridapDistributed: PVectorCounter
import GridapDistributed: DistributedSparseMatrixAssembler
import GridapDistributed: PVectorAllocationTrackOnlyValues
import GridapDistributed: ArrayAllocationTrackTouchedAndValues
import GridapDistributed: PVectorAllocationTrackTouchedAndValues
import GridapDistributed: DistributedCellDatum
import GridapDistributed: DistributedCellField
import GridapDistributed: DistributedFESpace
import GridapDistributed: DistributedSingleFieldFESpace
import GridapDistributed: DistributedSingleFieldFEFunction
import GridapDistributed: DistributedDomainContribution
import GridapDistributed: DistributedTriangulation
import GridapDistributed: DistributedMeasure
import GridapDistributed: DistributedMultiFieldFEFunction
import GridapDistributed: TransientDistributedCellField
import GridapDistributed: change_axes
import GridapDistributed: local_views

const OPTIONS_CG_JACOBI = "-pc_type jacobi -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_CG_AMG = "-pc_type gamg -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_MUMPS = "-pc_type lu -ksp_type preonly -ksp_converged_reason -pc_factor_mat_solver_type mumps"
const OPTIONS_MINRES = "-ksp_type minres -ksp_converged_reason -ksp_rtol 1.0e-10"

export OPTIONS_CG_JACOBI,OPTIONS_CG_AMG,OPTIONS_MUMPS,OPTIONS_NEUTON_MUMPS,OPTIONS_MINRES
export ParamJaggedArray
export ParamVectorAssemblyCache
export DistributedSnapshots
export DistributedRBSpace
export project_recast

include("DistributedUtils.jl")

export PMatrix
export MatrixAssemblyCache
include("PMatrix.jl")

export PVectorParamCounter
export DistributedParamAllocationVector
include("Algebra.jl")

export ParamJaggedArray
export ParamVectorAssemblyCache
export ParamJaggedArrayAssemblyCache
include("ParamJaggedArray.jl")

export ParamSubSparseMatrix
export AdjointPVector
export AdjointSubSparseMatrix
include("ParamSparseUtils.jl")

include("Primitives.jl")

export DistributedSingleFieldParamFESpace
export DistributedSingleFieldParamFEFunction
include("ParamFESpaces.jl")

include("TransientParamFESpaces.jl")

export DistributedMultiFieldParamFESpace
export DistributedMultiFieldParamFEFunction
export DistributedParamFESpace
include("MultiField.jl")

export DistributedContribution
export DistributedArrayContribution
export DistributedAffineContribution
export distributed_array_contribution
include("DistributedContribution.jl")

export DistributedTransientSnapshots
export DistributedSnapshots
include("DistributedSnapshots.jl")

export DistributedRBSpace
include("SingleRB.jl")

export load_distributed_snapshots
include("DistributedPostProcess.jl")
end
