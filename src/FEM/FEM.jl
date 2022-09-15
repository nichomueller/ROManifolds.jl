#= module FEM

include("../Utils/Utils.jl")
using .Utils
using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FEMSpaces
using Gridap.Fields
using Gridap.Geometry
using GridapGmsh
using Gridap.Helpers
using Gridap.Io
using Gridap.ReferenceFEs
using Gridap.TensorValues
import Gridap:∇
using Logging
using LineSearches:BackTracking

export assemble_FEM_structure
export get_FEMSpace
export LagrangianQuad
export lagrangianQuad
export LagrangianQuadRefFE
export Problem
export FEMProblem
export SteadyProblem
export UnsteadyProblem
export Info
export SteadyInfo
export UnsteadyInfo
export FEMSpacePoissonSteady
export FEMSpacePoissonUnsteady
export FEMSpaceStokesSteady
export FEMSpaceStokesUnsteady
export ProblemInfoSteady
export ProblemInfoUnsteady
export SteadyParametricInfo
export UnsteadyParametricInfo
export FEM_paths
export generate_dcube_discrete_model
export get_ParamInfo
export generate_vtk_file
export find_FE_elements
export generate_dcube_discrete_model

include("FEMSuperclasses.jl")
include("assemblers.jl")
include("FEMUtils.jl")
include("FEMSpaces.jl")
include("solvers.jl")

end =#

include("../Utils/Utils.jl")
include("FEM_Types.jl")
include("Lagrangian_Quad.jl")
include("FEM_Assemblers.jl")
include("FEM_Utils.jl")
include("FEM_Spaces.jl")
include("FEM_Solvers.jl")
