# general nonlinear case

function ODEs.allocate_odecache(
  ::ThetaMethod,
  odeop::ODEOperator,
  r0::TransientParamRealization,
  us0::NTuple{1,AbstractVector})

  u0 = us0[1]
  us0N = (u0,u0)
  odeopcache = allocate_odeopcache(odeop,r0,us0N)

  uθ = copy(u0)

  sysslvrcache = nothing
  odeslvrcache = (uθ,sysslvrcache)

  (odeslvrcache,odeopcache)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEOperator,
  r::TransientParamRealization,
  state0::NTuple{1,AbstractVector},
  odecache)

  u0 = state0[1]
  odeslvrcache,odeopcache = odecache
  uθ,sysslvrcache = odeslvrcache

  sysslvr = solver.sysslvr
  dt,θ = solver.dt,solver.θ

  x = statef[1]
  dtθ = θ*dt
  shift_time!(r,dtθ)
  function usx(x)
    copy!(uθ,u0)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end
  ws = (dtθ,1)

  update_odeopcache!(odeopcache,odeop,r)

  stageop = NonlinearParamStageOperator(odeop,odeopcache,r,usx,ws)

  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)

  shift_time!(r,dt*(1-θ))
  statef = ODEs._udate_theta!(statef,state0,dt,x)

  odeslvrcache = (uθ,sysslvrcache)
  odecache = (odeslvrcache,odeopcache)
  (r,statef,odecache)
end

function jacobian_and_residual(
  solver::ThetaMethod,
  odeop::ODEOperator,
  r::TransientParamRealization,
  state0::NTuple{1,AbstractVector},
  odecache)

  u0 = state0[1]
  odeslvrcache,odeopcache = odecache
  uθ, = odeslvrcache

  dt,θ = solver.dt,solver.θ

  x = copy(u0)
  dtθ = θ*dt
  shift_state!(x,1/dt)
  shift_time!(r,dt*(θ-1))
  function usx(x)
    copy!(uθ,u0)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end
  ws = (1,1/dtθ)

  update_odeopcache!(odeopcache,odeop,r)

  stageop = NonlinearParamStageOperator(odeop,odeopcache,r,usx,ws)
  A = jacobian(stageop,x)
  b = residual(stageop,x)

  return A,b
end

# linear case

function ODEs.allocate_odecache(
  ::ThetaMethod,
  odeop::ODEOperator{LinearParamODE},
  r0::TransientParamRealization,
  us0::NTuple{1,AbstractVector})

  u0 = us0[1]
  us0N = (u0,u0)
  odeopcache = allocate_odeopcache(odeop,r0,us0N)

  constant_stiffness = is_form_constant(odeop,0)
  constant_mass = is_form_constant(odeop,1)
  reuse = (constant_stiffness && constant_mass)

  A = allocate_jacobian(odeop,r0,us0N,odeopcache)
  b = allocate_residual(odeop,r0,us0N,odeopcache)

  sysslvrcache = nothing
  odeslvrcache = (reuse,A,b,sysslvrcache)

  (odeslvrcache,odeopcache)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEOperator{LinearParamODE},
  r::TransientParamRealization,
  state0::NTuple{1,AbstractVector},
  odecache)

  u0 = state0[1]
  odeslvrcache,odeopcache = odecache
  reuse,A,b,sysslvrcache = odeslvrcache

  sysslvr = solver.sysslvr
  dt,θ = solver.dt,solver.θ

  x = statef[1]
  fill!(x,zero(eltype(x)))
  dtθ = θ*dt
  shift_time!(r,dtθ)
  usx = (u0,x)
  ws = (dtθ,1)

  update_odeopcache!(odeopcache,odeop,r)

  stageop = LinearParamStageOperator(odeop,odeopcache,r,usx,ws,A,b,reuse,sysslvrcache)

  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)

  shift_time!(r,dt*(1-θ))
  statef = ODEs._udate_theta!(statef,state0,dt,x)

  odeslvrcache = (reuse,A,b,sysslvrcache)
  odecache = (odeslvrcache,odeopcache)
  (r,statef,odecache)
end

function jacobian_and_residual(
  solver::ThetaMethod,
  odeop::ODEOperator{LinearParamODE},
  r::TransientParamRealization,
  state0::NTuple{1,AbstractVector},
  odecache)

  u0 = state0[1]
  odeslvrcache,odeopcache = odecache
  reuse,A,b,sysslvrcache = odeslvrcache

  dt,θ = solver.dt,solver.θ

  x = copy(u0)
  fill!(x,zero(eltype(x)))
  dtθ = θ*dt
  shift_time!(r,dt*(θ-1))
  us = (x,x)
  ws = (1,1/dtθ)

  stageop = LinearParamStageOperator(odeop,odeopcache,r,us,ws,A,b,reuse,sysslvrcache)

  return stageop.A,stageop.b
end
