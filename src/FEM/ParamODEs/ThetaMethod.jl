# general nonlinear case

function allocate_odeparamcache(
  ::ThetaMethod,
  odeop::ODEParamOperator,
  r0::TransientRealization,
  us0::NTuple{1,AbstractVector})

  u0 = us0[1]
  us0N = (u0,u0)
  paramcache = allocate_paramcache(odeop,r0,us0N)

  uθ = copy(u0)

  sysslvrcache = nothing
  odeslvrcache = (uθ,sysslvrcache)

  (odeslvrcache,paramcache)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  odeparamcache)

  u0 = state0[1]
  odeslvrcache,paramcache = odeparamcache
  uθ,sysslvrcache = odeslvrcache

  sysslvr = solver.sysslvr
  dt,θ = solver.dt,solver.θ

  x = statef[1]
  dtθ = θ*dt
  shift!(r,dtθ)
  function usx(x)
    copy!(uθ,u0)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end
  ws = (dtθ,1)

  update_paramcache!(paramcache,odeop,r)

  stageop = NonlinearParamStageOperator(odeop,paramcache,r,usx,ws)

  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)

  shift!(r,dt*(1-θ))
  statef = ODEs._udate_theta!(statef,state0,dt,x)

  odeslvrcache = (uθ,sysslvrcache)
  odeparamcache = (odeslvrcache,paramcache)
  (r,statef,odeparamcache)
end

function Algebra.residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  state0::NTuple{1,AbstractVector})

  u0 = state0[1]

  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  shift!(r,dt*(θ-1))

  x = copy(u0)
  uθ = copy(u0)
  shift!(uθ,r,θ,1-θ)
  axpy!(dtθ,x,uθ)
  usx = (uθ,x)

  b = residual(odeop,r,usx)
  shift!(r,dt*(1-θ))

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  state0::NTuple{1,AbstractVector})

  u0 = state0[1]

  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  shift!(r,dt*(θ-1))

  x = copy(u0)
  uθ = copy(u0)
  shift!(uθ,r,θ,1-θ)
  axpy!(dtθ,x,uθ)
  usx = (uθ,x)
  ws = (1,1/dtθ)

  A = jacobian(odeop,r,usx,ws)
  shift!(r,dt*(1-θ))

  return A
end

# linear case

function allocate_odeparamcache(
  ::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r0::TransientRealization,
  us0::NTuple{1,AbstractVector})

  u0 = us0[1]
  us0N = (u0,u0)
  paramcache = allocate_paramcache(odeop,r0,us0N)

  constant_stiffness = is_form_constant(odeop,1)
  constant_mass = is_form_constant(odeop,2)

  A = allocate_jacobian(odeop,r0,us0N,paramcache)
  b = allocate_residual(odeop,r0,us0N,paramcache)

  sysslvrcache = nothing
  odeslvrcache = (A,b,sysslvrcache)

  (odeslvrcache,paramcache)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  odeparamcache)

  u0 = state0[1]
  odeslvrcache,paramcache = odeparamcache
  A,b,sysslvrcache = odeslvrcache

  sysslvr = solver.sysslvr
  dt,θ = solver.dt,solver.θ

  x = statef[1]
  fill!(x,zero(eltype(x)))
  dtθ = θ*dt
  shift!(r,dtθ)
  usx = (u0,x)
  ws = (dtθ,1)

  update_paramcache!(paramcache,odeop,r)

  stageop = LinearParamStageOperator(odeop,paramcache,r,usx,ws,A,b,sysslvrcache)

  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)

  shift!(r,dt*(1-θ))
  statef = ODEs._udate_theta!(statef,state0,dt,x)

  odeslvrcache = (A,b,sysslvrcache)
  odeparamcache = (odeslvrcache,paramcache)
  (r,statef,odeparamcache)
end

function Algebra.residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  state0::NTuple{1,AbstractVector})

  u0 = state0[1]

  dt,θ = solver.dt,solver.θ

  x = copy(u0)
  fill!(x,zero(eltype(x)))
  dtθ = θ*dt
  shift!(r,dt*(θ-1))
  us = (x,x)

  b = residual(odeop,r,us)
  shift!(r,dt*(1-θ))

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  state0::NTuple{1,AbstractVector})

  u0 = state0[1]

  dt,θ = solver.dt,solver.θ

  x = copy(u0)
  fill!(x,zero(eltype(x)))
  dtθ = θ*dt
  shift!(r,dt*(θ-1))
  us = (x,x)
  ws = (1,1/dtθ)

  A = jacobian(odeop,r,us,ws)
  shift!(r,dt*(1-θ))

  return A
end

# linear-nonlinear case

function allocate_odeparamcache(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  r0::TransientRealization,
  us0::NTuple{1,AbstractVector})

  odecache_lin = allocate_odeparamcache(solver,get_linear_operator(odeop),r0,us0)
  odecache_nlin = allocate_odeparamcache(solver,get_nonlinear_operator(odeop),r0,us0)
  return odecache_lin,odecache_nlin
end
