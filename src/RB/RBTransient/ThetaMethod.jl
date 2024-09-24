# general nonlinear case

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

# linear case

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

# utils

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
