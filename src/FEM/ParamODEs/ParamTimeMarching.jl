function stage_variable(solver::ODESolver,u0::AbstractVector)
  @notimplemented "For now, only theta methods are implemented"
end

function stage_variable(solver::ThetaMethod,u0::AbstractVector)
  return (copy(u0),)
end

function allocate_updatecache(solver::ODESolver,u0::AbstractVector)
  @notimplemented "For now, only theta methods are implemented"
end

function allocate_updatecache(solver::ThetaMethod,u0::AbstractVector)
  copy(u0)
end

function stage_weight(solver::ODESolver)
  @notimplemented "For now, only theta methods are implemented"
end

function stage_weight(solver::ThetaMethod)
  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  return (dtθ,1)
end

function ODEs.ode_start(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r0::TransientRealization,
  u0::AbstractVector)

  state0 = stage_variable(solver,u0)
  upcache = allocate_updatecache(solver,u0)
  order = get_order(odeop)
  us0 = tfill(u0,Val{order+1}())
  paramcache = allocate_paramcache(odeop,r0)
  syscache = allocate_systemcache(odeop,r0,us0,paramcache)
  return state0,(upcache,paramcache,syscache)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  state::NTuple{1,AbstractVector},
  odecache)

  u0 = state[1]
  x = statef[1]

  uθ,paramcache,syscache = odecache
  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  ws = (dtθ,1)

  shift!(r,dtθ)
  function state_update(x)
    copy!(uθ,u0)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end
  update_paramcache!(paramcache,odeop,r)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)
  shift!(r,dt*(1-θ))

  statef = ODEs._udate_theta!(statef,state,dt,x)
  return (r,statef)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  state::NTuple{1,AbstractVector},
  odecache)

  u0 = state[1]
  x = statef[1]
  fill!(x,zero(eltype(x)))

  uθ,paramcache,syscache = odecache
  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  ws = (dtθ,1)

  shift!(r,dtθ)
  state_update(x) = (u0,x)
  update_paramcache!(paramcache,odeop,r)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)
  shift!(r,dt*(1-θ))

  statef = ODEs._udate_theta!(statef,state,dt,x)
  return (r,statef)
end

function ODEs.ode_finish!(
  uf::AbstractVector,
  solver::ODESolver,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  statef::Tuple{Vararg{AbstractVector}},
  odecache)

  copy!(uf,first(statef))
  uf
end
