function solve_step!(
  uf::AbstractVector,
  op::ParamODEOperator,
  solver::θMethod,
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

function residual!(
  b::AbstractVector,
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  uθ = x
  vθ = (x-op.u0)/op.dtθ
  residual!(b,op.odeop,op.μ,op.tθ,(uθ,vθ),op.ode_cache)
end

function jacobian!(
  A::AbstractMatrix,
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  uF = x
  vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function allocate_residual(
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  allocate_residual(op.odeop,x,op.ode_cache)
end

function allocate_jacobian(
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  allocate_jacobian(op.odeop,x,op.ode_cache)
end

# MDEIM snapshots generation interface

function _evaluation_function(
  solver::θMethod,
  trial::Tsp,
  xh::AbstractMatrix,
  x0::AbstractVector) where Tsp

  times = get_times(solver)
  xhθ = solver.θ*xh + (1-solver.θ)*hcat(x0,xh[:,1:end-1])
  xh_t = _as_function(xh,times)
  xhθ_t = _as_function(xhθ,times)

  dtrial(t) = ∂t(trial(t))
  x_t(t) = EvaluationFunction(trial(t),xh_t(t))
  xθ_t(t) = EvaluationFunction(dtrial(t),xhθ_t(t))
  t -> TransientCellField(x_t(t),(xθ_t(t),))
end

_evaluation_function(u,args...) = u

function _evaluation_function(
  u::Gridap.ODEs.TransientFETools.TransientMultiFieldCellField,
  col::Int)

  u_col = Any[]
  for nf = eachindex(u.transient_single_fields)
    nf == col ? push!(u_col,u[col]) : push!(u_col,nothing)
  end
  u_col
end

function _vecdata_residual(
  op::ParamTransientFEOperator,
  solver::θMethod,
  sols::AbstractArray,
  params::AbstractArray,
  filter::Tuple{Vararg{Int}},
  args...;
  kwargs...)

  row,_ = filter
  test_row = get_test(op)[row]
  trial = get_trial(op)
  dv_row = _get_fe_basis(op.test,row)
  sol_μ = _as_function(sols,params)
  assem_row = SparseMatrixAssembler(test_row,test_row)
  op.assem = assem_row

  function vecdata(μ,t)
    u0 = get_free_dof_values(solver.uh0(μ))
    u = _evaluation_function(solver,trial(μ),sol_μ(μ),u0)
    collect_cell_vector(test_row,op.res(μ,t,u(t),dv_row),args...)
  end

  vecdata
end

function _matdata_jacobian(
  op::ParamTransientFEOperator,
  solver::θMethod,
  sols::AbstractArray,
  params::AbstractArray,
  filter::Tuple{Vararg{Int}},
  args...;
  kwargs...)

  row,col = filter
  test_row = get_test(op)[row]
  trial_col = get_trial(op)[col]
  dv_row = _get_fe_basis(op.test,row)
  du_col = _get_trial_fe_basis(get_trial(op)(nothing,nothing),col)
  sols_μ = _as_function(sols,params)
  assem_row_col = SparseMatrixAssembler(trial_col(nothing,nothing)[col],test_row)
  op.assem = assem_row_col

  γ = (1.0,1/(solver.dt*solver.θ))
  function matdata(μ,t)
    u0 = get_free_dof_values(solver.uh0(μ))
    u = _evaluation_function(solver,get_trial(op)(μ),sols_μ(μ),u0)
    u_col(t) = _evaluation_function(u(t),col)
    _matdata = ()
    for (i,γᵢ) in enumerate(γ)
      if γᵢ > 0.0
        _matdata = (_matdata...,
          collect_cell_matrix(
          trial_col(μ,t),
          test_row,
          γᵢ*op.jacs[i](μ,t,u_col(t),dv_row,du_col),
          args...))
      end
    end
    _vcat_matdata(_matdata)
  end

  matdata
end
