#

function RBSteady.allocate_rbcache(
  solver::ThetaMethod,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  rb_lhs_cache = allocate_jacobian(op,r)
  rb_rhs_cache = allocate_residual(op,r)
  syscache = (rb_lhs_cache,rb_rhs_cache)

  dt,θ = solver.dt,solver.θ
  shift!(r,dt*(θ-1))
  rbopcache = get_trial(op)(r)
  shift!(r,dt*(1-θ))

  return syscache,rbopcache
end

# general nonlinear case

function ODEs.allocate_odecache(
  solver::ThetaMethod,
  op::LinearNonlinearTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  odecache_lin = allocate_odecache(solver,get_linear_operator(op),r,us)
  odecache_nlin = allocate_odecache(solver,get_nonlinear_operator(op),r,us)
  odeslvrcache_nlin,odeopcache = odecache_nlin
  uθ, = odeslvrcache_nlin
  nlop = get_nonlinear_operator(op).op
  A_nlin = allocate_jacobian(nlop,r,us,odeopcache)
  b_nlin = allocate_residual(nlop,r,us,odeopcache)
  odeslvrcache_nlin = uθ,A_nlin,b_nlin
  odecache_nlin = odeslvrcache_nlin,odeopcache
  return (odecache_lin,odecache_nlin)
end

function get_stage_operator(
  solver::ThetaMethod,
  rbop::TransientRBOperator{LinearNonlinearParamODE},
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  odecache,
  rbcache)

  u0 = state0[1]

  # linear + nonlinear cache
  odecache_lin,odecache_nlin = odecache
  rbcache_lin,rbcache_nlin = rbcache

  # linear cache
  odeslvrcache_lin,odeopcache = odecache_lin
  reuse_lin,A_lin,b_lin,sysslvrcache_lin = odeslvrcache_lin
  rbsyscache_lin,rbopcache = rbcache_lin
  Â_lin,b̂_lin = rbsyscache_lin

  # nonlinear cache
  odeslvrcache_nlin,_ = odecache_nlin
  uθ,A_nlin,b_nlin = odeslvrcache_nlin
  rbop_nlin = get_nonlinear_operator(rbop)
  rbsyscache_nlin,_ = rbcache_nlin
  Â_nlin,b̂_nlin = rbsyscache_nlin
  rbsyscache_lin_nlin = (A_nlin,Â_nlin),(b_nlin,b̂_nlin)
  cache_lin_nlin = rbsyscache_lin_nlin,rbopcache

  # linear operator
  rbop_lin = get_linear_operator(rbop)
  lop = get_stage_operator(solver,rbop_lin,r,state0,odecache_lin,rbcache_lin)

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
  rbop_nlin = get_nonlinear_operator(rbop)
  stageop = RBNewtonRaphsonOperator(rbop_nlin,lop,odeopcache,r,us,ws,cache_lin_nlin)
  shift!(r,dt*(1-θ))

  return stageop
end

# linear case

function ODEs.allocate_odecache(
  solver::ThetaMethod,
  op::GenericTransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  dt,θ = solver.dt,solver.θ
  dtθ = θ*dt
  shift!(r,dt*(θ-1))

  (odeslvrcache,odeopcache) = allocate_odecache(solver,op.op,r,us)
  shift!(r,dt*(1-θ))

  return (odeslvrcache,odeopcache)
end

function get_stage_operator(
  solver::ThetaMethod,
  rbop::TransientRBOperator{LinearParamODE},
  r::TransientRealization,
  state0::NTuple{1,AbstractVector},
  odecache,
  rbcache)

  u0 = state0[1]
  odeslvrcache,odeopcache = odecache
  reuse,A,b,sysslvrcache = odeslvrcache
  rbsyscache,_ = rbcache
  Â,b̂ = rbsyscache

  dt,θ = solver.dt,solver.θ

  x = copy(u0)
  fill!(x,zero(eltype(x)))
  dtθ = θ*dt
  shift!(r,dt*(θ-1))
  us = (x,x)
  ws = (1,1/dtθ)

  stageop = LinearParamStageOperator(rbop,odeopcache,r,us,ws,(A,Â),(b,b̂),reuse,sysslvrcache)
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

function ParamDataStructures.shift!(a::BlockParamVector,r::TransientRealization,α::Number,β::Number)
  @inbounds for ai in blocks(a)
    ParamDataStructures.shift!(ai,r,α,β)
  end
end
