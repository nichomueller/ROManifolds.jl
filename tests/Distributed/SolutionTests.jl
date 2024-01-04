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
using PartitionedArrays

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
uf = copy(sol.u0)
w = copy(sol.u0)
dtθ = dt*θ
cache = nothing

tθ = t0+dtθ

if isnothing(cache)
  ode_cache = allocate_cache(feop,params,tθ)
  uf = copy(w)
  vθ = similar(w)
  vθ .= 0.0
  l_cache = nothing
end

ode_cache = update_cache!(ode_cache,feop,params,tθ)

lop = PTAffineThetaMethodOperator(feop,params,tθ,dtθ,w,ode_cache,vθ)

l_cache = solve!(uf,fesolver.nls,lop,l_cache)

uf = uf + w
if 0.0 < θ < 1.0
  @. uf = uf*(1.0/θ)-w*((1-θ)/θ)
end

# # 2nd iter ok
# w1 = copy(uf)
# uf1 = copy(uf)
# lop = PTAffineThetaMethodOperator(feop,params,tθ,dtθ,w1,ode_cache,vθ)
# b = l_cache.b
# A = l_cache.A
# ns = l_cache.ns
# residual!(b,lop,uf1)
# jacobian!(A,lop,uf1)
# numerical_setup!(ns,A)
# rmul!(b,-1)
# solve!(uf1,ns,b)
# uf1 = uf1 + w1
# if 0.0 < θ < 1.0
#   @. uf1 = uf1*(1.0/θ)-w1*((1-θ)/θ)
# end

# 2nd iter not ok
w .= uf
lop = PTAffineThetaMethodOperator(feop,params,tθ,dtθ,w,ode_cache,vθ)
l_cache = solve!(uf,fesolver.nls,lop,l_cache)
uf = uf + w
if 0.0 < θ < 1.0
  @. uf = uf*(1.0/θ)-w*((1-θ)/θ)
end

# gridap
np = 1
feop_t = get_feoperator_gridap(feop,params[np])
ode_op_t = get_algebraic_operator(feop_t)
ode_solver = ThetaMethod(LUSolver(),dt,θ)
w_t = zero_free_values(test)
sol_t = GenericODESolution(ode_solver,ode_op_t,w_t,t0,tf)
uf_t = copy(sol_t.u0)
w_t = copy(sol_t.u0)
cache_t = nothing

if isnothing(cache_t)
  ode_cache_t = TransientFETools.allocate_cache(ode_op_t)
  uf_t = copy(w_t)
  vθ_t = similar(w_t)
  vθ_t .= 0.0
  l_cache_t = nothing
  A_t,b_t = ODETools._allocate_matrix_and_vector(ode_op_t,tθ,w_t,ode_cache_t)
end

ode_cache_t = update_cache!(ode_cache_t,ode_op_t,tθ)

ODETools._matrix_and_vector!(A_t,b_t,ode_op_t,tθ,dtθ,w_t,ode_cache_t,vθ_t)

afop = AffineOperator(A_t,b_t)

newmatrix = true
l_cache_t = solve!(uf_t,ode_solver.nls,afop,l_cache_t,newmatrix)

uf_t = uf_t + w_t
if 0.0 < θ < 1.0
  uf_t = uf_t*(1.0/θ)-w_t*((1-θ)/θ)
end

# # 2nd iter
# w1_t = copy(uf_t)
# uf1_t = copy(uf_t)
# ODETools._matrix_and_vector!(A_t,b_t,ode_op_t,tθ,dtθ,w1_t,ode_cache_t,vθ_t)
# afop = AffineOperator(A_t,b_t)
# A_t = afop.matrix
# b_t = afop.vector
# ns_t = l_cache_t
# if newmatrix
#   numerical_setup!(ns_t,A_t)
# end
# solve!(uf1_t,ns_t,b_t)
# uf1_t = uf1_t + w1_t
# if 0.0 < θ < 1.0
#   uf1_t = uf1_t*(1.0/θ)-w1_t*((1-θ)/θ)
# end

# 2nd iter not ok
w_t .= uf_t
ODETools._matrix_and_vector!(A_t,b_t,ode_op_t,tθ,dtθ,w_t,ode_cache_t,vθ_t)
afop = AffineOperator(A_t,b_t)
l_cache_t = solve!(uf_t,ode_solver.nls,afop,l_cache_t,newmatrix)
uf_t = uf_t + w_t
if 0.0 < θ < 1.0
  uf_t = uf_t*(1.0/θ)-w_t*((1-θ)/θ)
end

# # comparison
map(local_views(uf),local_views(uf_t)) do uf,uf_t
  @check uf[1] ≈ uf_t
end

map(local_views(uf1),local_views(uf1_t)) do uf1,uf1_t
  @check uf1[1] ≈ uf1_t
end

map(own_values(w),own_values(w_t)) do uf,uf_t
  @check uf[1] ≈ uf_t
end

map(local_views(A),local_views(A_t)) do A,A_t
  @check A[1] ≈ A_t
end

map(local_views(b),local_views(b_t)) do b,b_t
  @check b[1] ≈ b_t
end

map(local_views(uf),local_views(uf_t)) do uf,uf_t
  println(uf[1] - uf_t)
end
