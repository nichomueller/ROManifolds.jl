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

using Mabla
using Mabla.FEM

using SingleFieldUtilsTests

import Gridap.ODEs.TransientFETools: get_algebraic_operator,GenericODESolution

ntimes = 3
t0 = 0
dt = 0.1
tf = (ntimes-1)*dt
θ = 0.5
nparams = 2
params = realization(feop,nparams)

ode_op = get_algebraic_operator(feop)
fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
w = get_free_dof_values(uh0μ(μ))
sol = PODESolution(fesolver,ode_op,μ,w,t0,tf)

results = PTArray[]
for (uh,t) in sol
  push!(results,copy(uh))
end

for np in 1:nparams
  feop_t = get_feoperator_gridap(feop,params[np])
  ode_op_t = get_algebraic_operator(feop_t)
  ode_solver = ThetaMethod(LUSolver(),dt,θ)
  sol_t = GenericODESolution(ode_solver,ode_op_t,w[np],t0,tf)

  results_t = Vector{Float}[]
  for (uh,t) in sol_gridap
    push!(results_t,copy(uh))
  end

  for (α,β) in zip(results,results_t)
    @check isapprox(α[n],β)
  end
end

end # module
