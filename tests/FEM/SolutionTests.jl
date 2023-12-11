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

# function Gridap.ODEs.ODETools.solve_step!(uf::AbstractVector,
#   solver::ThetaMethod,
#   op::AffineODEOperator,
#   u0::AbstractVector,
#   t0::Real,
#   cache) # -> (uF,tF)

#   dt = solver.dt
#   solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
#   tθ = t0+dtθ

#   if cache === nothing
#   ode_cache = allocate_cache(op)
#   vθ = similar(u0)
#   vθ .= 0.0
#   l_cache = nothing
#   A, b = Gridap.ODEs.ODETools._allocate_matrix_and_vector(op,t0,u0,ode_cache)
#   else
#   ode_cache, vθ, A, b, l_cache = cache
#   end

#   ode_cache = update_cache!(ode_cache,op,tθ)

#   Gridap.ODEs.ODETools._matrix_and_vector!(A,b,op,tθ,dtθ,u0,ode_cache,vθ)
#   afop = AffineOperator(A,b)

#   # println("Norm ok residual: $(norm(b))")
#   println("Norm ok jacobian: $(norm(A))")
#   newmatrix = true
#   l_cache = solve!(uf,solver.nls,afop,l_cache,newmatrix)
#   println("Norm ok uf (pre): $(norm(uf))")

#   uf = uf + u0
#   if 0.0 < solver.θ < 1.0
#     uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
#   end
#   println("Norm ok uf (post): $(norm(uf))")
#   cache = (ode_cache, vθ, A, b, l_cache)

#   tf = t0+dt
#   return (uf,tf,cache)

# end
function Gridap.ODEs.TransientFETools.jacobians!(
  A::AbstractMatrix,
  op::TransientFETools.TransientFEOperatorsFromWeakForm,
  t::Real,
  xh::TransientFETools.TransientCellField,
  γ::Tuple{Vararg{Real}},
  cache)
  println("Pre norm jac ok: $(norm(A))")
  _matdata_jacobians = TransientFETools.fill_jacobians(op,t,xh,γ)
  matdata = Gridap.ODEs.TransientFETools._vcat_matdata(_matdata_jacobians)
  assemble_matrix_add!(A,op.assem_t, matdata)
  A
end
function Gridap.ODEs.TransientFETools._matdata_jacobian(
  op::TransientFETools.TransientFEOperatorsFromWeakForm,
  t::Real,
  xh::T,
  i::Integer,
  γᵢ::Real) where T
  println("Jac ok $i at time $t, coeff $γᵢ")
  println("dv ok: $(norm(xh.cellfield.dirichlet_values)), fv ok: $(norm((xh.cellfield.free_values)))")
  Uh = evaluate(get_trial(op),nothing)
  V = get_test(op)
  du = get_trial_fe_basis(Uh)
  v = get_fe_basis(V)
  matdata = collect_cell_matrix(Uh,V,γᵢ*op.jacs[i](t,xh,du,v))
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

# w = get_free_dof_values(uh0μ(params))
# w .= 1
# u0 = copy(w)
# uf = copy(w)
# tθ = t0+dt*θ
# μ = params
# ode_cache = allocate_cache(feop,μ,tθ)
# vθ = similar(u0)
# vθ .= 0.0
# nl_cache = nothing
# ode_cache = update_cache!(ode_cache,feop,μ,tθ)
# lop = PTAffineThetaMethodOperator(feop,μ,tθ,dt*θ,u0,ode_cache,vθ)
# A = allocate_jacobian(lop,uf)
# LinearAlgebra.fillstored!(A,1.)
# vθ = lop.vθ
# Xh, = lop.ode_cache
# xh = TransientCellField(EvaluationFunction(Xh[1],vθ),(EvaluationFunction(Xh[1],vθ),))
# v = get_fe_basis(test)
# du = get_trial_fe_basis(trial0)
# _matdata = TransientFETools.fill_jacobians(feop,lop.μ,lop.t,xh,(1.,1/(θ*dt)))
# matdata = Gridap.ODEs.TransientFETools._vcat_matdata(_matdata)
# assemble_matrix_add!(A,feop.assem,matdata)

# #Gridap
# u0_t = copy(w[1])
# uf_t = copy(w[1])
# feop_t = get_feoperator_gridap(feop,params[1])
# op = get_algebraic_operator(feop_t)
# ode_cache_t = allocate_cache(op)
# vθ_t = similar(u0_t)
# vθ_t .= 0.0
# A_t,b_t = ODETools._allocate_matrix_and_vector(op,t0,u0,ode_cache)
# LinearAlgebra.fillstored!(A_t,1.)
# ode_cache_t = update_cache!(ode_cache_t,op,tθ)
# Xh_t, = ode_cache_t
# xh_t = TransientCellField(EvaluationFunction(Xh_t[1],vθ_t),(EvaluationFunction(Xh_t[1],vθ_t),))
# _matdata_t = TransientFETools.fill_jacobians(feop_t,tθ,xh_t,(1.,1/(θ*dt)))
# matdata_t = Gridap.ODEs.TransientFETools._vcat_matdata(_matdata_t)
# assemble_matrix_add!(A_t,feop_t.assem_t,matdata_t)

# @assert A_t ≈ A[1]
# check_ptarray(matdata[1][1],matdata_t[1][1];n = 1)
# @check matdata[2] == matdata_t[2]
# @check matdata[3] == matdata_t[3]

# w = get_free_dof_values(uh0μ(params))
# w .= 1
# u0 = copy(w)
# uf = copy(w)
# tθ = t0+dt*θ
# μ = params
# ode_cache = allocate_cache(feop,μ,tθ)
# vθ = similar(u0)
# vθ .= 0.0
# nl_cache = nothing
# ode_cache = update_cache!(ode_cache,feop,μ,tθ)
# lop = PTAffineThetaMethodOperator(feop,μ,tθ,dt*θ,u0,ode_cache,vθ)
# # @which solve!(uf,solver.nls,lop,nl_cache)
# # @which b = residual(lop,uf)
# b = allocate_residual(lop,uf)
# b .= 1.
# # @which residual!(b,lop,uf)
# vθ = lop.vθ
# # @which residual!(b,lop,(uf,vθ))
# Xh, = lop.ode_cache
# xh = TransientCellField(EvaluationFunction(Xh[1],uf),(EvaluationFunction(Xh[1],vθ),))
# # @which residual!(b,lop.feop,lop.μ,lop.t,xh)
# v = get_fe_basis(test)
# dc = integrate(feop.res(lop.μ,lop.t,xh,v))
# vecdata = collect_cell_vector(test,dc)
# assemble_vector_add!(b,feop.assem,vecdata)

# #Gridap
# u0_t = copy(w[1])
# uf_t = copy(w[1])
# feop_t = get_feoperator_gridap(feop,params[1])
# op = get_algebraic_operator(feop_t)
# ode_cache_t = allocate_cache(op)
# vθ_t = similar(u0_t)
# vθ_t .= 0.0
# _,b_t = ODETools._allocate_matrix_and_vector(op,t0,u0,ode_cache)
# b_t .= 1.
# ode_cache_t = update_cache!(ode_cache_t,op,tθ)
# # @which residual!(b,op,tθ,(u0_t,vθ_t),ode_cache_t)
# Xh_t, = ode_cache_t
# xh_t = TransientCellField(EvaluationFunction(Xh_t[1],uf_t),(EvaluationFunction(Xh_t[1],vθ_t),))
# # @which residual!(b_t,feop_t,tθ,xh_t,ode_cache_t)
# vecdata_t = collect_cell_vector(test,feop_t.res(tθ,xh_t,v))
# assemble_vector_add!(b_t,feop_t.assem_t,vecdata_t)

# @assert b_t ≈ b[1]
end # module
