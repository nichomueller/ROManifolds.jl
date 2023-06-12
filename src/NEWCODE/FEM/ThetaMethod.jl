function Gridap.ODEs.TransientFETools.solve_step!(
  uf::AbstractVector,
  solver::θMethod,
  op::ParamODEOperator,
  μ::AbstractVector,
  u0::AbstractVector,
  t0::Real,
  cache)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op)
    vθ = similar(u0)
    nl_cache = nothing
  else
    ode_cache,vθ,nl_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  nlop = ParamThetaMethodNonlinearOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < solver.θ < 1.0
    uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,nl_cache)

  tf = t0+dt
  return (uf,tf,cache)
end

"""
Nonlinear operator that represents the θ-method nonlinear operator at a
given time step, i.e., A(t,u_n+θ,(u_n+θ-u_n)/dt)
"""
struct ParamThetaMethodNonlinearOperator <: NonlinearOperator
  odeop::ParamODEOperator
  μ::AbstractVector
  tθ::Float64
  dtθ::Float64
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function Gridap.ODEs.TransientFETools.residual!(
  b::AbstractVector,
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  uθ = x
  vθ = (x-op.u0)/op.dtθ
  residual!(b,op.odeop,op.μ,op.tθ,(uθ,vθ),op.ode_cache)
end

function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  uF = x
  vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function Gridap.ODEs.ODETools.allocate_residual(
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  allocate_residual(op.odeop,x,op.ode_cache)
end

function Gridap.ODEs.ODETools.allocate_jacobian(
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  allocate_jacobian(op.odeop,x,op.ode_cache)
end

function zero_initial_guess(op::ParamThetaMethodNonlinearOperator)
  x0 = similar(op.u0)
  fill!(x0,zero(eltype(x0)))
  x0
end

# MDEIM snapshots generation interface

function _evaluation_function(
  solver::θMethod,
  trial::TransientTrialFESpace,
  xh::AbstractMatrix,
  x0=zeros(size(xh,1)))

  times = get_times(solver)
  xhθ = solver.θ*xh + (1-solver.θ)*hcat(x0,xh[:,1:end-1])
  xh_t = _as_function(xh,times)
  xhθ_t = _as_function(xhθ,times)

  dtrial(t) = ∂t(trial(t))
  x_t(t) = EvaluationFunction(trial(t),xh_t(t))
  xθ_t(t) = EvaluationFunction(dtrial(t),xhθ_t(t))
  t -> TransientCellField(x_t(t),(xθ_t(t),))
end

function _vecdata_residual(
  solver::θMethod,
  op::ParamTransientFEOperator,
  sols::AbstractMatrix,
  params::Table)

  trial = get_trial(op)
  test = get_test(op)
  dv = get_fe_basis(test)
  sol_μ = _as_function(sols,params)
  u(μ,t) = get_evaluation_function(solver,trial(μ),sol_μ(μ))(t)                 # add initial condition if needed
  (μ,t) -> collect_cell_vector(test,op.res(μ,t,(u(μ,t),uθ(μ,t)),dv))
  # vecdatum(μ,t) = collect_cell_vector(test,op.res(μ,t,(u(μ,t),uθ(μ,t)),dv))
  # vecdata = pmap(μ -> map(t -> vecdatum(μ,t),times),params,uh)

  # b = allocate_residual(op,first(first(sols)),nothing,filter)
  # pmap(d -> assemble_vector!(b,op.assem,d,filter),vecdata...)
end

function Gridap.ODEs.TransientFETools._matdata_jacobian(
  solver::θMethod,
  op::ParamTransientFEOperator,
  sols::AbstractMatrix,
  params::Table)

  trial = get_trial(op)
  test = get_test(op)
  dv = get_fe_basis(test)
  du = get_trial_fe_basis(trial(nothing,nothing))
  sol_μ = _as_function(sols,params)
  u(μ,t) = get_evaluation_function(solver,trial(μ),sol_μ(μ))(t)                 # add initial condition if needed
  (μ,t) -> collect_cell_matrix(trial(μ,t),test,op.jac(μ,t,u(μ,t),dv,du))
  # matdatum(μ,t) = collect_cell_matrix(trial(μ,t),test,op.jac(μ,t,u(μ,t),dv,du))
  # matdata = pmap(μ -> map(t -> matdatum(μ,t),times),params,uh)

  # A = allocate_jacobian(op,first(first(sols)),nothing,filter)
  # pmap(d -> assemble_jacobian!(A,op.assem,d,filter),matdata...)
end

# _matdata_jacobians = fill_jacobians(op,μ,t,uh,γ)
# matdata = _vcat_matdata(_matdata_jacobians)
# assemble_matrix_add!(A,op.assem,matdata)
