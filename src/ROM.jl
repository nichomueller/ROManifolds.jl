module ROM

export compute_speedup
export compute_error
export compute_relative_error
export ∂₁, ∂₂, ∂₃

include("FEM/Utils/Utils.jl")
using ROM.Utils

export OrderedFESpace
export CartesianFESpace
export OrderedFEFunction
export slow_index
export fast_index
export get_dof_map
export get_sparse_dof_map
export flatten

include("FEM/DofMaps/DofMaps.jl")
using ROM.DofMaps

export TProductModel
export TProductFESpace

include("FEM/TProduct/TProduct.jl")
using ROM.TProduct

export Realization
export TransientRealization
export UniformSampling
export NormalSampling
export HaltonSampling
export ParamSpace
export TransientParamSpace
export ParamFunction
export TransientParamFunction
export realization

export ParamArray
export ConsecutiveParamArray
export ParamSparseMatrix
export BlockParamArray
export Snapshots
export select_snapshots

include("FEM/ParamDataStructures/ParamDataStructures.jl")
using ROM.ParamDataStructures

include("FEM/ParamAlgebra/ParamAlgebra.jl")
using ROM.ParamAlgebra

include("FEM/ParamReferenceFEs/ParamReferenceFEs.jl")
using ROM.ParamReferenceFEs

include("FEM/ParamGeometry/ParamGeometry.jl")
using ROM.ParamGeometry

export ParamTrialFESpace
export TrialParamFESpace
export MultiFieldParamFESpace

include("FEM/ParamFESpaces/ParamFESpaces.jl")
using ROM.ParamFESpaces

export FEDomains
export ParamFEOperator
export LinearParamFEOperator
export LinearNonlinearParamFEOperator

include("FEM/ParamSteady/ParamSteady.jl")
using ROM.ParamSteady

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace
export TransientParamFEOperator
export TransientParamLinearFEOperator
export LinearNonlinearTransientParamFEOperator

include("FEM/ParamODEs/ParamODEs.jl")
using ROM.ParamODEs

export Reduction
export PODReduction
export TTSVDReduction
export SupremizerReduction
export MDEIMReduction
export AdaptiveReduction

export RBSolver
export solution_snapshots
export residual_snapshots
export jacobian_snapshots

export reduction
export tpod
export ttsvd
export gram_schmidt
export orth_projection

export Projection
export PODProjection
export TTSVDProjection
export NormedProjection
export BlockProjection
export ReducedProjection
export projection
export get_basis
export get_cores
export project
export inv_project
export galerkin_projection
export union_bases
export contraction

export RBSpace
export reduced_spaces

export empirical_interpolation
export reduced_jacobian
export reduced_residual
export reduced_weak_form

export RBOperator
export reduced_operator

export ROMPerformance
export create_dir
export eval_performance
export plot_a_solution
export load_snapshots
export load_operator
export load_results

include("RB/RBSteady/RBSteady.jl")
using ROM.RBSteady

export TransientReduction
export TransientMDEIMReduction
export TransientProjection
export TransientRBOperator

include("RB/RBTransient/RBTransient.jl")
using ROM.RBTransient
end
