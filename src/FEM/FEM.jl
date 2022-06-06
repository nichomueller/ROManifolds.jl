"""
The exported names are
$(EXPORTS)
"""
module FEM

using Mabla.Utils
using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
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

export get_FESpace
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
export FESpacePoissonSteady
export FESpacePoissonUnsteady
export FESpaceStokesSteady
export FESpaceStokesUnsteady
export ProblemSpecificsSteady
export ProblemSpecificsUnteady
export ParametricSpecificsSteady
export ParametricSpecificsUnSteady

export FEM_paths
export generate_dcube_discrete_model
export get_Parametric_specifics
export generate_vtk_file
export find_FE_elements
export generate_dcube_discrete_model

include("assemblers.jl")
include("FEMSuperclasses.jl")
include("FEMUtils.jl")
include("FEMSpaces.jl")
include("solvers.jl")

end
