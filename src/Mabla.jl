module Mabla
include("FEM/FEM.jl")
using Mabla.FEM

export ParamRealization
export TransientParamRealization
export UniformSampling
export NormalSampling
export ParamSpace
export TransientParamSpace
export ParamFunction, 𝑓ₚ
export TransientParamFunction, 𝑓ₚₜ
export realization
export get_params
export get_times

export AbstractParamContainer
export ParamContainer

export ParamArray
export ParamVector
export ParamMatrix
export array_of_similar_arrays
export zero_param_array
export get_at_offsets
export recenter

export ParamBlockArray

export ParamReindex
export PosNegParamReindex

export ParamField
export ParamFieldGradient
export GenericParamField
export OperationParamField

export SingleFieldParamFESpace
export FESpaceToParamFESpace
export length_dirichlet_values
export length_free_values

export MultiFieldParamFESpace

export TrialParamFESpace
export TrialParamFESpace!
export HomogeneousTrialParamFESpace

export ParamFEFunction
export SingleFieldParamFEFunction
export MultiFieldParamFEFunction

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace

export ParamCounter
export ParamInserterCSC
export ParamCSSR

export TransientParamFEOperator
export TransientParamFEOpFromWeakForm
export LinearTransientParamFEOperator
export NonlinearTransientParamFEOperator
export residual_for_trian!
export jacobian_for_trian!

export ODEParamOperator
export ODEParamOpFromTFEOp

export ODEParamSolution
export GenericODEParamSolution

export TransientParamFESolution

export ParamString
export ParamVisualizationData

export get_order
export get_L2_norm_matrix
export get_H1_norm_matrix

# include("RB/RB.jl")
# using Mabla.RB

# include("Distributed/Distributed.jl")
# using Mabla.Distributed
end # module
