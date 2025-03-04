module ParamAlgebra

using LinearAlgebra
using BlockArrays
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using Gridap.ReferenceFEs

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers

using ROManifolds.ParamDataStructures

import ArraysOfArrays: innersize
import ROManifolds.DofMaps: OIdsToIds, add_ordered_entries!
import UnPack: @unpack

include("ParamAlgebraInterfaces.jl")

export ParamNonlinearOperator
export GenericParamNonlinearOperator
export LazyParamNonlinearOperator
export AbstractParamCache
export ParamOpCache
export LazyParamOpCache
export ParamOpSysCache
export lazy_residual
export lazy_residual!
export allocate_lazy_residual
export lazy_jacobian
export lazy_jacobian!
export lazy_jacobian_add!
export allocate_lazy_jacobian
export allocate_paramcache
export update_paramcache!
export next_index!
export allocate_systemcache
include("NonlinearParamOperators.jl")

include("ParamSolvers.jl")

include("ParamIterativeSolvers.jl")

end # module
