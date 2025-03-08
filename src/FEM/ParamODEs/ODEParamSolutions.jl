"""
    struct ODEParamSolution{V} <: ODESolution
      solver::ODESolver
      stageop::NonlinearParamOperator
      u0::AbstractVector
    end
"""
struct ODEParamSolution{V} <: ODESolution
  solver::ODESolver
  stageop::NonlinearParamOperator
  u0::AbstractVector
end

initial_realization(sol::ODEParamSolution) = initial_realization(sol.stageop)
initial_condition(sol::ODEParamSolution) = sol.u0

ParamDataStructures.num_params(sol::ODEParamSolution) = num_params(sol.stageop)
ParamDataStructures.num_times(sol::ODEParamSolution) = num_times(sol.stageop)

function Base.iterate(sol::ODEParamSolution)
  # initialize
  timeid = 0
  nlop = sol.stageop[timeid]
  state0,odecache = ode_start(sol.solver,nlop,sol.u0)

  # march
  statef = copy.(state0)
  rf,statef = ode_march!(statef,sol.solver,nlop,state0,odecache)

  # finish
  timeid += 1
  uf = copy(sol.u0)
  uf = ode_finish!(uf,sol.solver,nlop,statef,odecache)

  state = (timeid,statef,state0,uf,odecache)
  return uf,state
end

function Base.iterate(sol::ODEParamSolution,state)
  timeid,state0,statef,uf,odecache = state
  nlop = sol.stageop[timeid]

  if timeid >= get_final_time(nlop) - eps()
    return nothing
  end

  statef = ode_march!(statef,sol.solver,nlop,state0,odecache)

  uf = ode_finish!(uf,sol.solver,nlop,statef,odecache)

  state = (timeid,statef,state0,uf,odecache)
  return uf,state
end

function Base.collect(sol::ODEParamSolution{V}) where V
  values = _collect_param_solutions(sol)
  t = @timed values = _collect_param_solutions(sol)
  tracker = CostTracker(t;name="FEM time marching",nruns=num_params(sol))
  return values,tracker
end

function Algebra.solve(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealization,
  u0::AbstractVector)

  nlop = ode_parameterize(solver,odeop,r,u0)
  ODEParamSolution(solver,nlop)
end

function Algebra.solve(
  solver::ODESolver,
  odeop::SplitODEParamOperator,
  r::TransientRealization,
  u0::AbstractVector)

  solve(solver,set_domains(odeop),r,u0)
end

# function Algebra.solve(
#   solver::ODESolver,
#   odeop::LinearNonlinearParamFEOperator,
#   r::TransientRealization,
#   u0::AbstractVector)

#   TransientParamFESolution(solver,join_operators(odeop),r,u0)
# end

function Algebra.solve(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealization,
  uh0::Function)

  params = get_params(r)
  u0 = get_free_dof_values(uh0(params))
  solve(solver,odeop,r,u0)
end

# utils

function _collect_param_solutions(sol::ODEParamSolution{<:ConsecutiveParamVector{T}}) where T
  u0item = testitem(sol.u0)
  s = size(u0item,1),num_params(sol)*num_times(sol)
  values = similar(u0item,T,s)
  for (k,uk) in enumerate(sol)
    _collect_solutions!(values,uk,k)
  end
  return ConsecutiveParamArray(values)
end

function _collect_solutions!(
  values::AbstractMatrix,
  uk::ConsecutiveParamVector,
  k::Int)

  datak = get_all_data(uk)
  nparams = param_length(uk)
  for ip in 1:nparams
    itp = (it-1)*nparams+ip
    for is in axes(values,1)
      @inbounds v = datak[is,ip]
      @inbounds values[is,itp] = v
    end
  end
end
