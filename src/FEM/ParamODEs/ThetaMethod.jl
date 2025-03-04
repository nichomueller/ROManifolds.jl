# general nonlinear case

function allocate_odeparamcache(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r0::TransientRealization,
  us0::NTuple{1,AbstractVector})

  u0 = us0[1]
  us0N = (u0,u0)
  uθ = copy(u0)
  paramcache = allocate_paramcache(odeop,r0,us0N)
  sysslvrcache = nothing
  (uθ,paramcache,sysslvrcache)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  cache)

  u0 = state0[1]
  uθ,paramcache,sysslvrcache = cache
  sysslvr = solver.sysslvr
  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  ws = (dtθ,1)
  x = statef[1]

  # update
  shift!(r,dtθ)
  function us(x)
    copy!(uθ,u0)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end
  update_paramcache!(paramcache,odeop,r)
  stageop = ParamStageOperator(odeop,paramcache,r,us,ws)
  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)
  statef = ODEs._udate_theta!(statef,state0,dt,x)
  shift!(r,dt*(1-θ))

  cache = (uθ,paramcache,sysslvrcache)
  (r,statef,cache)
end

function Algebra.residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  state::NTuple{1,AbstractParamVector},
  u0::AbstractParamVector)

  u = state[1]
  dt,θ = solver.dt,solver.θ
  x = copy(u)
  uθ = copy(u)

  shift!(r,dt*(θ-1))
  shift!(uθ,u0,θ,1-θ)
  shift!(x,u0,1/dt,-1/dt)
  us = (uθ,x)
  b = residual(odeop,r,us)
  shift!(r,dt*(1-θ))

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  state::NTuple{1,AbstractVector},
  u0::AbstractParamVector)

  u = state[1]
  dt,θ = solver.dt,solver.θ
  x = copy(u)
  uθ = copy(u)

  shift!(r,dt*(θ-1))
  shift!(uθ,u0,θ,1-θ)
  shift!(x,u0,1/dt,-1/dt)
  us = (uθ,x)
  ws = (1,1)

  A = jacobian(odeop,r,us,ws)
  shift!(r,dt*(1-θ))

  return A
end

# linear case

function allocate_odeparamcache(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r0::TransientRealization,
  us0::NTuple{1,AbstractVector})

  u0 = us0[1]
  us0N = (u0,u0)
  ws = (1,1)

  paramcache = allocate_paramcache(odeop,r0,us0N)
  A,b = allocate_systemcache(odeop,r0,us0N,ws,paramcache)
  odeparamcache = SystemCache(paramcache,A,b)
  sysslvrcache = nothing

  (odeparamcache,sysslvrcache)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  cache)

  u0 = state0[1]
  odeparamcache,sysslvrcache = cache
  paramcache = odeparamcache.paramcache

  sysslvr = solver.sysslvr
  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  x = statef[1]
  fill!(x,zero(eltype(x)))
  ws = (dtθ,1)

  # update
  shift!(r,dtθ)
  us(x) = (u0,x)
  update_paramcache!(paramcache,odeop,r)
  stageop = ParamStageOperator(odeop,odeparamcache,r,us,ws)
  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)
  statef = ODEs._udate_theta!(statef,state0,dt,x)
  shift!(r,dt*(1-θ))

  cache = (odeparamcache,sysslvrcache)
  (r,statef,cache)
end

function Algebra.residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  state::NTuple{1,AbstractParamVector},
  u0::AbstractParamVector)

  u = state[1]
  dt,θ = solver.dt,solver.θ
  x = copy(u)
  fill!(x,zero(eltype(x)))
  us = (x,x)

  shift!(r,dt*(θ-1))
  b = residual(odeop,r,us)
  shift!(r,dt*(1-θ))

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  state::NTuple{1,AbstractVector},
  u0::AbstractParamVector)

  u = state[1]
  dt,θ = solver.dt,solver.θ
  ws = (1,1)
  x = copy(u)
  fill!(x,zero(eltype(x)))
  us = (x,x)

  shift!(r,dt*(θ-1))
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

  u0 = us0[1]
  us0N = (u0,u0)
  uθ = copy(u0)

  lop = get_linear_operator(odeop)
  paramcache,sysslvrcache = allocate_odeparamcache(solver,lop,r0,us0)
  A,b = allocate_systemcache(lop,r0,us0,paramcache)
  odeparamcache = SystemCache(paramcache,A,b)

  (uθ,odeparamcache,sysslvrcache)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  cache)

  u0 = state0[1]
  uθ,odeparamcache,sysslvrcache = cache
  paramcache = odeparamcache.paramcache

  sysslvr = solver.sysslvr
  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  x = statef[1]
  fill!(x,zero(eltype(x)))
  ws = (dtθ,1)

  # update
  shift!(r,dtθ)
  function us(x)
    copy!(uθ,u0)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end
  update_paramcache!(paramcache,odeop,r)
  stageop = ParamStageOperator(odeop,odeparamcache,r,us,ws)
  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)
  statef = ODEs._udate_theta!(statef,state0,dt,x)
  shift!(r,dt*(1-θ))

  cache = (uθ,odeparamcache,sysslvrcache)
  (r,statef,cache)
end


# utils

function ParamDataStructures.shift!(
  a::ConsecutiveParamVector,
  a0::ConsecutiveParamVector,
  α::Number,
  β::Number)

  data = get_all_data(a)
  data0 = get_all_data(a0)
  data′ = copy(data)
  np = param_length(a0)
  for ipt = param_eachindex(a)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        data[is,ipt] = α*data[is,ipt] + β*data0[is,ipt]
      end
    else
      for is in axes(data,1)
        data[is,ipt] = α*data[is,ipt] + β*data′[is,ipt-np]
      end
    end
  end
end

function ParamDataStructures.shift!(
  a::BlockParamVector,
  a0::BlockParamVector,
  α::Number,
  β::Number)

  @inbounds for (ai,a0i) in zip(blocks(a),blocks(a0))
    ParamDataStructures.shift!(ai,a0i,α,β)
  end
end
