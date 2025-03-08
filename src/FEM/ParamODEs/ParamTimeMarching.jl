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

  function us(x)
    copy!(uθ,u0)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end

  ParamStageOperator(odeop,r,us,ws)
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

  us(x) = (u0,x0)

  ParamStageOperator(odeop,r,us,ws)
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

  state0 = (copy(u0),)
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
  dt,θ = solver.dt,solver.θ

  shift!(nlop.r,θ*dt)
  update_paramcache!(nlop.paramcache,nlop.op,nlop.r)
  solve!(x,solver.sysslvr,nlop,syscache)
  shift!(nlop.r,dt*(1-θ))

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
