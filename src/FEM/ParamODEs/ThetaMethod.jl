# general nonlinear case

function ODEs.allocate_odecache(
  ::ThetaMethod,
  odeop::ODEParamOperator,
  r0::TransientRealization,
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
  odeop::ODEParamOperator,
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  odecache)

  u0 = state0[1]
  odeslvrcache,odeopcache = odecache
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

  update_odeopcache!(odeopcache,odeop,r)

  stageop = NonlinearParamStageOperator(odeop,odeopcache,r,usx,ws)

  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)

  shift!(r,dt*(1-θ))
  statef = ODEs._udate_theta!(statef,state0,dt,x)

  odeslvrcache = (uθ,sysslvrcache)
  odecache = (odeslvrcache,odeopcache)
  (r,statef,odecache)
end

# linear case

function ODEs.allocate_odecache(
  ::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r0::TransientRealization,
  us0::NTuple{1,AbstractVector})

  u0 = us0[1]
  us0N = (u0,u0)
  odeopcache = allocate_odeopcache(odeop,r0,us0N)

  constant_stiffness = is_form_constant(odeop,1)
  constant_mass = is_form_constant(odeop,2)
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
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
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
  shift!(r,dtθ)
  usx = (u0,x)
  ws = (dtθ,1)

  update_odeopcache!(odeopcache,odeop,r)

  stageop = LinearParamStageOperator(odeop,odeopcache,r,usx,ws,A,b,reuse,sysslvrcache)

  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)

  shift!(r,dt*(1-θ))
  statef = ODEs._udate_theta!(statef,state0,dt,x)

  odeslvrcache = (reuse,A,b,sysslvrcache)
  odecache = (odeslvrcache,odeopcache)
  (r,statef,odecache)
end

# linear-nonlinear case

function ODEs.allocate_odecache(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  r0::TransientRealization,
  us0::NTuple{1,AbstractVector})

  odecache_lin = allocate_odecache(solver,get_linear_operator(odeop),r0,us0)
  odecache_nlin = allocate_odecache(solver,get_nonlinear_operator(odeop),r0,us0)
  return odecache_lin,odecache_nlin
end
