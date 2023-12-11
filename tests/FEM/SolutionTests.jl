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

using Mabla
using Mabla.FEM

using Main.SingleFieldUtilsFEMTests

import Gridap.Helpers: @check
import Gridap.ODEs.TransientFETools: get_algebraic_operator,GenericODESolution

function Gridap.ODEs.ODETools._vector!(b,odeop,tθ,dtθ,u0,ode_cache,vθ)
  # Us,=ode_cache
  # println("Norm x: $(norm(u0))")
  # println("Norm x0: $(norm(vθ))")
  # println("Norm dv: $(norm(Us[1].dirichlet_values))")
  residual!(b,odeop,tθ,(u0,vθ),ode_cache)
  b .*= -1.0
  println("Norm b: $(norm(b[1]))")
end

ntimes = 3
tf = (ntimes-1)*dt
nparams = 2
params = realization(feop,nparams)

fesolver = PThetaMethod(LUSolver(),uh0μ,θ,dt,t0,tf)
w = get_free_dof_values(uh0μ(params))
sol = PODESolution(fesolver,feop,params,w,t0,tf)

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
  for (uh,t) in sol_t
    push!(results_t,copy(uh))
  end

  for (α,β) in zip(results,results_t)
    @check isapprox(α[np],β) "Detected difference in value for index $np"
  end
end

# results[1][1]
# results[1][2]

# np = 1
# feop_t = get_feoperator_gridap(feop,params[np])
# ode_op_t = get_algebraic_operator(feop_t)
# ode_solver = ThetaMethod(LUSolver(),dt,θ)
# sol_t = GenericODESolution(ode_solver,ode_op_t,w[np],t0,tf)

# results_t = Vector{Float}[]
# for (uh,t) in sol_t
#   push!(results_t,copy(uh))
# end

# results_t[1]
# results_t[2]
# results_t[2] - results[1][2]
# maximum(abs.(results_t[1] - results[1][1]))
# maximum(abs.(results_t[2] - results[1][2]))

# u0 = PTArray([ones(size(w[1])) for _ = 1:nparams])
# uf = copy(w)
# μ = params
# dtθ = dt*θ
# tθ = t0+dtθ
# cache = nothing
# #
# ode_cache = allocate_cache(feop,μ,tθ)
# vθ = similar(u0)
# vθ .= 0.0
# l_cache = nothing
# ode_cache = update_cache!(ode_cache,feop,μ,tθ)
# lop = PTAffineThetaMethodOperator(feop,μ,tθ,dt*θ,u0,ode_cache,vθ)
# l_cache = solve!(uf,fesolver.nls,lop,l_cache)
# uf .+= u0
# @. uf = uf*(1.0/θ)-u0*((1-θ)/θ)
# cache = (ode_cache,vθ,l_cache)
# u0 = uf

# # gridap
# np = 1
# u0_t = ones(size(w[1]))
# uf_t = copy(w[1])
# feop_t = get_feoperator_gridap(feop,params[np])
# ode_op_t = get_algebraic_operator(feop_t)
# ode_solver = ThetaMethod(LUSolver(),dt,θ)
# cache_t = nothing
# #
# ode_cache_t = TransientFETools.allocate_cache(ode_op_t)
# vθ_t = similar(u0_t)
# vθ_t .= 0.0
# l_cache_t = nothing
# A_t,b_t = ODETools._allocate_matrix_and_vector(ode_op_t,t0,u0_t,ode_cache_t)
# ode_cache_t = TransientFETools.update_cache!(ode_cache_t,ode_op_t,tθ)
# ODETools._matrix_and_vector!(A_t,b_t,ode_op_t,tθ,dtθ,u0_t,ode_cache_t,vθ_t)
# afop_t = AffineOperator(A_t,b_t)
# l_cache_t = solve!(uf_t,ode_solver.nls,afop_t,l_cache_t,true)
# uf_t = uf_t + u0_t
# uf_t = uf_t*(1.0/θ)-u0_t*((1-θ)/θ)
# cache_t = (ode_cache_t,vθ_t,A_t,b_t,l_cache_t)
# u0_t = uf_t

# Us, = ode_cache_t
# x0 = zeros(size(w[1]))
# x1 = ones(size(w[1]))
# u = TransientCellField(EvaluationFunction(Us[1],x1),(EvaluationFunction(Us[2],x0),))
# r(v) = ∫(v*∂t(u))dΩ + ∫(a(μ[1],tθ)*∇(v)⋅∇(u))dΩ - ∫(f(μ[1],tθ)*v)dΩ - ∫(h(μ[1],tθ)*v)dΓn
# manb = -assemble_vector(r,test)
# manb ≈ b_t
# manb ≈ l_cache.b[1]

# Us, = ode_cache
# x0 = zero(w)
# x1 = copy(w)
# x1 .= 1.
# xh = TransientCellField(EvaluationFunction(Us[1],x1),(EvaluationFunction(Us[2],x0),))
# dc = integrate(feop.res(μ,tθ,xh,get_fe_basis(test)))
# vecdata = collect_cell_vector(test,dc)
# b = copy(w)
# assemble_vector_add!(b,feop.assem,vecdata)
# rmul!(b,-1)
# manb ≈ b[1]
# bph
# # try 1
# Us, = ode_cache_t
# x0 = zeros(size(w[1]))
# x1 = ones(size(w[1]))
# u = TransientCellField(EvaluationFunction(Us[1],x1),(EvaluationFunction(Us[2],x0),))
# r(v) = ∫(v*∂t(u))dΩ + ∫(a(μ[1],tθ)*∇(v)⋅∇(u))dΩ - ∫(f(μ[1],tθ)*v)dΩ - ∫(h(μ[1],tθ)*v)dΓn
# manb = -assemble_vector(r,test)
# manb ≈ b_t

# # try 2
# function temp_solve!(
#   x::AbstractVector,
#   ls::LinearSolver,
#   op::NonlinearOperator,
#   cache::Nothing)
#   Mx = PTArray([M*x[1],M*x[2]])
#   fill!(x,zero(eltype(x)))
#   b = residual(op, x) - Mx
#   A = jacobian(op, x)
#   ss = symbolic_setup(ls, A)
#   ns = numerical_setup(ss,A)
#   rmul!(b,-1)
#   solve!(x,ns,b)
#   Algebra.LinearSolverCache(A,b,ns)
# end

# function temp_solve!(
#   x::AbstractVector,
#   ls::LinearSolver,
#   op::NonlinearOperator,
#   cache)
#   Mx = PTArray([M*x[1],M*x[2]])
#   fill!(x,zero(eltype(x)))
#   b = cache.b
#   A = cache.A
#   ns = cache.ns
#   residual!(b, op, x)
#   b -= Mx
#   numerical_setup!(ns,A)
#   rmul!(b,-1)
#   solve!(x,ns,b)
#   cache
# end

# u0 = w
# uf = copy(w)
# μ = params
# dtθ = dt*θ
# tθ = t0+dtθ
# cache = nothing
# M = assemble_matrix((du,dv)->∫(dv*du)dΩ,trial(μ[1],dt),test)/dtθ
# #
# if isnothing(cache)
#   ode_cache = allocate_cache(feop,μ,tθ)
#   vθ = similar(u0)
#   vθ .= 0.0
#   l_cache = nothing
# else
#   ode_cache,vθ,l_cache = cache
# end
# ode_cache = update_cache!(ode_cache,feop,μ,tθ)
# lop = PTAffineThetaMethodOperator(feop,μ,tθ,dt*θ,u0,ode_cache,vθ)
# l_cache = temp_solve!(uf,fesolver.nls,lop,l_cache)
# if 0.0 < θ < 1.0
#   @. uf = uf*(1.0/θ)-u0*((1-θ)/θ)
# end
# cache = (ode_cache,vθ,l_cache)
# u0 = uf

# ################################################################################
# using UnPack
# cache = nothing
# u0 = ones(size(w[1]))
# uf = copy(w[1])
# odeop = get_algebraic_operator(feop_t)
# solve_step!(uf,ode_solver,odeop,u0,t0,cache)
# cfeop_t = TransientFETools.TransientFEOperatorFromWeakForm{ConstantMatrix}(feop_t.res,feop_t.rhs,
#   feop_t.jacs,feop_t.assem_t,feop_t.trials,feop_t.test,feop_t.order)
# codeop = get_algebraic_operator(feop_t)
# _uf = copy(w[1])
# solve_step!(_uf,ode_solver,codeop,u0,t0,cache)
# @assert _uf ≈ uf
end # module
