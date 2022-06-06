"""
The exported names are
$(EXPORTS)
"""
module RB
using Mabla.Utils
using Mabla.FEM
using Arpack

#= export M_DEIM_POD
export M_DEIM_offline
export M_DEIM_online
export MDEIM_offline
export DEIM_offline
export MDEIM_offline_algebraic
export MDEIM_offline_functional
export DEIM_offline_algebraic
export DEIM_offline_functional
export get_snaps_MDEIM
export get_snaps_DEIM =#

export RBProblem
export RBSteadyProblem
export RBUnsteadyProblem
export PoissonSteady
export PoissonSGRB
export PoissonSPGRB
export PoissonUnsteady
export PoissonSTGRB
export PoissonSTPGRB
export StokesUnsteady
export StokesSTGRB
export ROMSpecificsSteady
export ROMSpecificsUnsteady
export setup
export offline_phase
export online_phase
export ROM_paths

include("M_DEIM.jl")
include("MV_snapshots.jl")
include("RBPoisson_steady.jl")
include("RBPoisson_unsteady.jl")
include("RBStokes_unsteady.jl")
include("RBSuperclasses.jl")
include("RBUtils.jl")
include("S-GRB_Poisson.jl")
include("S-PGRB_Poisson.jl")
include("ST-GRB_Poisson.jl")
include("ST-PGRB_Poisson.jl")
include("ST-GRB_Stokes.jl")
end
