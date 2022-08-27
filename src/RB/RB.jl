#= module RB
include("../Utils/Utils.jl")
include("../FEM/FEM.jl")
using .Utils
using .FEM
using Arpack
using LinearAlgebra
using Plots
using SparseArrays

#= import assemble_FEM_structure
import get_FEMSpace
import LagrangianQuad
import lagrangianQuad
import LagrangianQuadRefFE
import Problem
import FEMProblem
import SteadyProblem
import UnsteadyProblem
import Info
import SteadyInfo
import UnsteadyInfo
import FEMSpacePoissonSteady
import FEMSpacePoissonUnsteady
import FEMSpaceStokesSteady
import FEMSpaceStokesUnsteady
import ProblemInfoSteady
import ProblemInfoUnsteady
import SteadyParametricInfo
import UnsteadyParametricInfo
import FEM_paths
import generate_dcube_discrete_model
import get_ParamInfo
import generate_vtk_file
import find_FE_elements
import generate_dcube_discrete_model =#

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
export ROMInfoSteady
export ROMInfoUnsteady
export setup
export offline_phase
export online_phase
export ROM_paths

include("RBSuperclasses.jl")
include("RBUtils.jl")
include("MV_snapshots.jl")
include("M_DEIM.jl")
include("RBPoisson_steady.jl")
include("RBPoisson_unsteady.jl")
include("RBStokes_unsteady.jl")
include("S-GRB_Poisson.jl")
include("S-PGRB_Poisson.jl")
include("ST-GRB_Poisson.jl")
include("ST-PGRB_Poisson.jl")
include("ST-GRB_Stokes.jl")
end =#

include("RBSuperclasses.jl")
include("RBUtils.jl")
include("post_process.jl")
include("M_DEIM.jl")
