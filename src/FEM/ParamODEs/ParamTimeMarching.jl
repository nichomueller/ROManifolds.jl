function stage_variable(solver::ODESolver,u0::AbstractVector)
  @notimplemented "For now, only theta methods are implemented"
end

function stage_variable(solver::ThetaMethod,u0::AbstractVector)
  return (copy(u0),)
end

function stage_weight(solver::ODESolver)
  @notimplemented "For now, only theta methods are implemented"
end

function stage_weight(solver::ThetaMethod)
  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  return (dtθ,1)
end

function ode_parameterize(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealization,
  u0::AbstractVector)

  @notimplemented "For now, only theta methods are implemented"
end

function ode_parameterize(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  u0::AbstractVector)

  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  ws = (dtθ,1)

  function state_update(x)
    copy!(uθ,u0)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end
  shift = ShiftRules(solver,state_update)

  ParamStageOperator(odeop,r,shift,ws)
end

function ode_parameterize(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  u0::AbstractVector)

  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  ws = (dtθ,1)

  x0 = copy(u0)
  fill!(x0,zero(eltype(x0)))

  state_update(x) = (u0,x0)
  shift = ShiftRules(solver,state_update)

  ParamStageOperator(odeop,r,shift,ws)
end

function allocate_odesystemcache(
  solver::ODESolver,
  nlop::ParamStageOperator,
  u0::AbstractVector)

  @notimplemented "For now, only theta methods are implemented"
end

function allocate_odesystemcache(
  solver::ThetaMethod,
  nlop::ParamStageOperator,
  u0::AbstractVector)

  uθ = copy(u0)
  syscache = allocate_systemcache(nlop,u0)
  (uθ,syscache)
end

function ODEs.ode_start(
  solver::ThetaMethod,
  nlop::ParamStageOperator,
  u0::AbstractVector)

  state0 = stage_variable(solver,u0)
  odesyscache = allocate_odesystemcache(solver,nlop,u0)
  return state0,odesyscache
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  nlop::ParamStageOperator,
  state0::NTuple{1,AbstractVector},
  odecache)

  x = statef[1]
  uθ,syscache = odecache
  dt = solver.dt

  solve!(x,solver.sysslvr,nlop,syscache)

  statef = ODEs._udate_theta!(statef,state0,dt,x)
  return statef
end

function ODEs.ode_finish!(
  uf::AbstractVector,
  solver::ODESolver,
  nlop::ParamStageOperator,
  statef::Tuple{Vararg{AbstractVector}},
  odecache)

  copy!(uf,first(statef))
  uf
end
