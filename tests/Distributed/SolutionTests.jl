include("./SingleFieldUtilsFEMTests.jl")

module SolutionTests

using LinearAlgebra

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

using Mabla
using Mabla.FEM

using Main.SingleFieldUtilsFEMTests

import Gridap.Helpers: @check
import Gridap.ODEs.TransientFETools: get_algebraic_operator,GenericODESolution

ntimes = 3
tf = (ntimes-1)*dt
nparams = 2
params = realization(feop,nparams)

fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
w = get_free_dof_values(uh0μ(params))
sol = PODESolution(fesolver,feop,params,w,t0,tf)

results = []
for (uh,t) in sol
  push!(results,copy(uh))
end

for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  ode_op_t = get_algebraic_operator(feop_t)
  ode_solver = ThetaMethod(LUSolver(),dt,θ)
  w_t = zero_free_values(test)
  sol_t = GenericODESolution(ode_solver,ode_op_t,w_t,t0,tf)

  results_t = []
  for (uh,t) in sol_t
    push!(results_t,copy(uh))
  end

  for (α,β) in zip(results,results_t)
    map(local_views(α),local_views(β)) do α,β
      @check isapprox(α[np],β) "Detected difference in value for index $np"
    end
  end
end
end # module
