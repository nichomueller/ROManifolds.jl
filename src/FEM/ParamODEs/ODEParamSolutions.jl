"""
    struct ODEParamSolution{V} <: ODESolution
      solver::ODESolver
      stageop::NonlinearParamOperator
      u0::V
    end
"""
struct ODEParamSolution{V} <: ODESolution
  solver::ODESolver
  stageop::NonlinearParamOperator
  u0::V
end

ParamDataStructures.num_params(sol::ODEParamSolution) = num_params(sol.stageop)
ParamDataStructures.num_times(sol::ODEParamSolution) = num_times(sol.stageop)

function Base.iterate(sol::ODEParamSolution)
  # initialize
  nlopit = iterate(sol.stageop)
  if isnothing(nlopit)
    return nothing
  end
  nlop,nlopstate = nlopit
  state0,odecache = ode_start(sol.solver,nlop,sol.u0)

  # march
  statef = copy.(state0)
  statef = ode_march!(statef,sol.solver,nlop,state0,odecache)

  # finish
  uf = copy(sol.u0)
  uf = ode_finish!(uf,sol.solver,nlop,statef,odecache)

  state = (nlopstate,statef,state0,uf,odecache)
  return uf,state
end

function Base.iterate(sol::ODEParamSolution,state)
  nlopstate,state0,statef,uf,odecache = state

  nlopit = iterate(sol.stageop,nlopstate)
  if isnothing(nlopit)
    return nothing
  end
  nlop,nlopstate = nlopit

  # march
  statef = ode_march!(statef,sol.solver,nlop,state0,odecache)

  # finish
  uf = ode_finish!(uf,sol.solver,nlop,statef,odecache)

  state = (nlopstate,statef,state0,uf,odecache)
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
  ODEParamSolution(solver,nlop,u0)
end

function Algebra.solve(
  solver::ODESolver,
  odeop::SplitODEParamOperator,
  r::TransientRealization,
  u0::AbstractVector)

  solve(solver,set_domains(odeop),r,u0)
end

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

initial_condition(sol::ODEParamSolution) = sol.u0

function _collect_param_solutions(sol)
  @notimplemented
end

function _collect_param_solutions(sol::ODEParamSolution{<:ConsecutiveParamVector{T}}) where T
  u0item = testitem(sol.u0)
  ncols = num_params(sol)*num_times(sol)
  values = similar(u0item,T,(size(u0item,1),ncols))
  for (k,uk) in enumerate(sol)
    _collect_solutions!(values,uk,k)
  end
  return ConsecutiveParamArray(values)
end

function _collect_param_solutions(sol::ODEParamSolution{<:BlockParamVector{T}}) where T
  u0item = testitem(sol.u0)
  ncols = num_params(sol)*num_times(sol)
  values = mortar(map(b -> similar(b,T,(size(b,1),ncols)),blocks(u0item)))
  for (k,uk) in enumerate(sol)
    for i in 1:blocklength(u0item)
      _collect_solutions!(blocks(values)[i],blocks(uk)[i],k)
    end
  end
  return BlockParamArray(map(ConsecutiveParamArray,blocks(values)))
end

function _collect_solutions!(
  values::AbstractMatrix,
  ui::ConsecutiveParamVector,
  it::Int)

  datai = get_all_data(ui)
  nparams = param_length(ui)
  for ip in 1:nparams
    itp = (it-1)*nparams+ip
    for is in axes(values,1)
      @inbounds v = datai[is,ip]
      @inbounds values[is,itp] = v
    end
  end
end
