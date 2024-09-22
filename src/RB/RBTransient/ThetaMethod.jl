# general nonlinear case

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

function get_stage_operator(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  cache)

  u0 = state0[1]
  odeslvrcache,odeopcache = cache
  uθ,= odeslvrcache

  x = copy(u0)
  fill!(x,zero(eltype(x)))

  dt,θ = solver.dt,solver.θ

  dtθ = θ*dt
  shift!(r,dt*(θ-1))

  function us(u)
    copy!(uθ,u)
    shift!(uθ,r,θ,1-θ)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end
  ws = (1,1/dtθ)

  stageop = NonlinearParamStageOperator(odeop,odeopcache,r,us,ws)
  shift!(r,dt*(1-θ))

  return stageop
end

function Algebra.solve!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  cache)

  x = statef[1]
  sysslvr = solver.sysslvr
  (odeslvrcache,odeopcache),rbcache = cache
  uθ,sysslvrcache = odeslvrcache

  stageop = get_stage_operator(solver,odeop,r,state0,cache)
  solve!(x,sysslvr,stageop,sysslvrcache)
  return x
end

# linear case

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

function get_stage_operator(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  cache)

  u0 = state0[1]
  (odeslvrcache,odeopcache),rbcache = cache
  reuse,A,b,sysslvrcache = odeslvrcache
  Â,b̂ = rbcache

  dt,θ = solver.dt,solver.θ

  x = copy(u0)
  fill!(x,zero(eltype(x)))
  dtθ = θ*dt
  shift!(r,dt*(θ-1))
  us = (x,x)
  ws = (1,1/dtθ)

  stageop = LinearParamStageOperator(odeop,odeopcache,r,us,ws,(A,Â),(b,b̂),reuse,sysslvrcache)
  shift!(r,dt*(1-θ))

  return stageop
end

function Algebra.solve!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  cache)

  x = statef[1]
  sysslvr = solver.sysslvr
  (odeslvrcache,odeopcache),rbcache = cache
  reuse,A,b,sysslvrcache = odeslvrcache

  stageop = get_stage_operator(solver,odeop,r,state0,cache)
  solve!(x,sysslvr,stageop,sysslvrcache)
  return x
end

# linear-nonlinear case

function Algebra.solve!(
  staterb::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  r::TransientRealization,
  statefe::NTuple{1,AbstractVector},
  odecache)

  x̂ = staterb[1]
  x = statefe[1]
  odecache_lin,odecache_nlin = odecache
  odeslvrcache_nlin,odeopcache_nlin = odecache_nlin

  lop = get_linear_operator(odeop)
  nlop = get_nonlinear_operator(odeop)

  stageop_lin = get_stage_operator(solver,lop,r,x,odecache_lin)
  A_lin = jacobian(stageop_lin,x)
  b_lin = residual(stageop_lin,x)
  sysslvrcache = ((A_lin,A_nlin),(b_lin,b_nlin))
  sysslvr = solver.sysslvr

  stageop = get_stage_operator(solver,odeop,r,statefe,odecache_nlin)
  solve!(x̂,x,sysslvr,stageop,sysslvrcache)
  return x̂
end

function Algebra.solve!(
  x̂::AbstractVector,
  x::AbstractVector,
  nls::NewtonRaphsonSolver,
  stageop::NonlinearParamStageOperator,
  cache::Tuple)

  A_cache,b_cache = cache
  A_lin, = A_cache
  trial = get_trial(stageop.odeop)(stageop.rx)

  b = residual!(b_cache,stageop,x)
  A = jacobian!(A_cache,stageop,x)
  b .+= A_lin*x̂

  dx̂ = similar(b)
  ss = symbolic_setup(nls.ls,A)
  ns = numerical_setup(ss,A)

  nonlinear_rb_solve!(x̂,x,A,b,A_cache,b_cache,dx̂,ns,nls,stageop,trial)
end

function ParamDataStructures.shift!(a::AbstractParamArray,r::TransientRealization,α::Number,β::Number)
  b = copy(a)
  np = num_params(r)
  @assert param_length(a) == param_length(r)
  for i = param_eachindex(a)
    it = slow_index(i,np)
    if it > 1
      a[i] .= α*a[i] + β*b[i-np]
    end
  end
end

function ParamDataStructures.shift!(a::BlockVectorOfVectors,r::TransientRealization,α::Number,β::Number)
  @inbounds for ai in blocks(a)
    ParamDataStructures.shift!(ai,r,α,β)
  end
end
