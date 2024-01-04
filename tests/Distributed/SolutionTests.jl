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

# results = []
# for (uh,t) in sol
#   push!(results,copy(uh))
# end

np = 1
feop_t = get_feoperator_gridap(feop,params[np])
ode_op_t = get_algebraic_operator(feop_t)
ode_solver = ThetaMethod(LUSolver(),dt,θ)
w_t = zero_free_values(test)
sol_t = GenericODESolution(ode_solver,ode_op_t,w_t,t0,tf)

# results_t = []
# for (uh,t) in sol_t
#   push!(results_t,copy(uh))
# end

# for (r,r_t) in zip(results,results_t)
#   map(local_views(r),local_views(r_t)) do r,r_t
#     println(r[1])
#     println(r_t)
#   end
# end

uf = copy(sol.u0)
w = copy(sol.u0)
dtθ = dt*θ
cache = nothing

# uf,tf,l_cache = solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,l_cache)

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

# # residual!(b,ode_op,params,tθ,(w,vθ),ode_cache)
# Xh, = ode_cache
# # xh = TransientCellField(EvaluationFunction(Xh[1],w),(EvaluationFunction(Xh[2],vθ),))
# # GridapDistributed._EvaluationFunction(EvaluationFunction,Xh[1],w,false)
# # free_values = change_ghost(w,Xh[1].gids,is_consistent=false,make_consistent=true)
# same_partition = (w.index_partition === partition(Xh[1].gids))
# a_new = same_partition ? w : change_ghost(T,w,Xh[1].gids)
# consistent!(a_new) |> wait

# l_cache = solve!(uf,fesolver.nls,lop,l_cache)
x = uf
b = residual(lop,x)
A = jacobian(lop,x)
ss = symbolic_setup(fesolver.nls,A)
ns = numerical_setup(ss,A)
rmul!(b,-1)
solve!(x,ns,b)
Algebra.LinearSolverCache(A,b,ns)

uf = uf + w
if 0.0 < θ < 1.0
  @. uf = uf*(1.0/θ)-w*((1-θ)/θ)
end

w .= uf
cache = l_cache

# gridap
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

# # ODETools._vector!(b_t,ode_op_t,tθ,dtθ,w_t,ode_cache_t,vθ_t)
# # residual!(b_t,ode_op_t,tθ,(w_t,vθ_t),ode_cache_t)
# Xh_t, = ode_cache_t
# # xh_t = TransientCellField(EvaluationFunction(Xh[1],w_t),(EvaluationFunction(Xh[2],vθ_t),))
# # GridapDistributed._EvaluationFunction(EvaluationFunction,Xh[1],w_t,false)
# import GridapDistributed: change_ghost
# # free_values = change_ghost(w_t,Xh_t[1].gids,is_consistent=false,make_consistent=true)
# same_partition_t = (w_t.index_partition === partition(Xh_t[1].gids))
# a_new_t = same_partition_t ? w_t : change_ghost(T,w_t,Xh_t[1].gids)
# consistent!(a_new_t) |> wait

afop = AffineOperator(A_t,b_t)

newmatrix = true
l_cache_t = solve!(uf_t,ode_solver.nls,afop,l_cache_t,newmatrix)

uf_t = uf_t + w_t
if 0.0 < θ < 1.0
  uf_t = uf_t*(1.0/θ)-w_t*((1-θ)/θ)
end

w_t .= uf_t
cache_t = l_cache_t

# # comparison
map(local_views(w),local_views(w_t)) do uf,uf_t
  @check uf[1] ≈ uf_t
end

map(own_values(w),own_values(w_t)) do uf,uf_t
  @check uf[1] ≈ uf_t
end

# testing consistent!
import GridapDistributed: change_ghost

w = copy(sol.u0)
w .= pi
a_new = w
consistent!(a_new) |> wait

w_t = copy(sol_t.u0)
w_t .= pi
a_new_t = w_t
consistent!(a_new_t) |> wait

map(local_views(w),local_views(w_t)) do w,w_t
  @check w[1] ≈ w_t
end
