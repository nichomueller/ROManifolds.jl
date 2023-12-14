module Distributed
using Mabla.Utils
using Mabla.FEM
using Mabla.RB

using LinearAlgebra
using SparseArrays
using PartitionedArrays
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
import Gridap.Helpers: @notimplemented

import PartitionedArrays: is_consistent
import PartitionedArrays: exchange_impl!
import PartitionedArrays: allocate_local_values

import GridapDistributed: DistributedCellDatum
import GridapDistributed: DistributedCellField
import GridapDistributed: DistributedFESpace
import GridapDistributed: DistributedSingleFieldFESpace
import GridapDistributed: DistributedSingleFieldFEFunction
import GridapDistributed: DistributedSparseMatrixAssembler
import GridapDistributed: DistributedDomainContribution
import GridapDistributed: DistributedTriangulation
import GridapDistributed: DistributedMeasure
import GridapDistributed: DistributedMultiFieldPTFEFunction
import GridapDistributed: TransientDistributedCellField
import GridapDistributed: local_views

const OPTIONS_CG_JACOBI = "-pc_type jacobi -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_CG_AMG = "-pc_type gamg -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_MUMPS = "-pc_type lu -ksp_type preonly -ksp_converged_reason -pc_factor_mat_solver_type mumps"
const OPTIONS_MINRES = "-ksp_type minres -ksp_converged_reason -ksp_rtol 1.0e-10"

export OPTIONS_CG_JACOBI,OPTIONS_CG_AMG,OPTIONS_MUMPS,OPTIONS_NEUTON_MUMPS,OPTIONS_MINRES
export PTJaggedArray
export PTVectorAssemblyCache

include("PTArray.jl")
include("Primitives.jl")
include("FESpaces.jl")
include("MultiField.jl")
include("PTCellData.jl")
include("Transient.jl")
end
