module Mabla

include("Utils/Utils.jl")
using Mabla.Utils

export Float
export get_parent_dir
export create_dir
export correct_path
export save
export load
export num_active_dirs
export vec_to_mat_idx
export slow_idx
export fast_idx
export index_pairs
export change_mode
export compress_array
export recenter
export tpod
export gram_schmidt!
export orth_complement!
export orth_projection

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
export allocate_param_array
export zero_param_array
export get_at_offsets
export recenter

export ParamBlockArray

export ParamReindex
export PosNegParamReindex

export ParamField
export ParamFieldGradient
export GenericParamField
export ZeroParamField
export ConstantParamField
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
export TransientMultiFieldTrialParamFESpace

export ParamCounter
export ParamInserterCSC
export ParamCSSR

export TransientParamFEOperator
export TransientParamFEOperatorFromWeakForm
export AffineTransientParamFEOperator
export NonlinearTransientParamFEOperator
export residual_for_trian!
export jacobian_for_trian!

export ODEParamOperator
export ConstantODEParamOperator
export ConstantMatrixODEParamOperator
export AffineODEParamOperator
export NonlinearODEParamOperator
export ODEParamOpFromFEOp

export ThetaMethodParamOperator
export AffineThetaMethodParamOperator

export ODEParamSolution
export GenericODEParamSolution

export TransientParamFESolution

export ParamString
export ParamVisualizationData

export get_order
export get_L2_norm_matrix
export get_H1_norm_matrix

include("RBNew/RB.jl")
using Mabla.RB

include("Distributed/Distributed.jl")
using Mabla.Distributed
end # module
