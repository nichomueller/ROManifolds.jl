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
import FEMProblemS
import FEMProblemST
import Info
import InfoS
import InfoST
import FEMSpacePoissonS
import FEMSpacePoissonST
import FEMSpaceStokesS
import FEMSpaceStokesST
import ProblemInfoSteady
import ProblemInfoUnsteady
import ParamInfoS
import ParamInfoST
import FEM_paths
import generate_dcube_discrete_model
import ParamInfo
import generate_vtk_file
import find_FE_elements
import generate_dcube_discrete_model =#

export RBProblem
export RBProblemS
export RBProblemST
export PoissonS
export PoissonSGRB
export PoissonSPGRB
export PoissonST
export PoissonSTGRB
export PoissonSTPGRB
export StokesST
export StokesSTGRB
export ROMInfoS
export ROMInfoST
export setup
export offline_phase
export online_phase
export ROM_paths

include("RBSuperclasses.jl")
include("RBUtils.jl")
include("MV_snapshots.jl")
include("MDEIM.jl")
include("RBPoisson_steady.jl")
include("RBPoisson_unsteady.jl")
include("RBStokes_unsteady.jl")
include("S-GRB_Poisson.jl")
include("S-PGRB_Poisson.jl")
include("ST-GRB_Poisson.jl")
include("ST-PGRB_Poisson.jl")
include("ST-GRB_Stokes.jl")
end =#

include("RB_Types.jl")
include("RB_Utils.jl")
include("Post_Process.jl")
include("MDEIM.jl")
