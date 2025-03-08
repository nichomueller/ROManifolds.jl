function Algebra.allocate_residual(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b,
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  @abstractmethod
end

function Algebra.residual(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}})

  paramcache = allocate_paramcache(nlop,r;evaluated=true)
  residual(nlop,r,us,paramcache)
end

function Algebra.residual(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  b = allocate_residual(nlop,r,us,paramcache)
  residual!(b,nlop,r,us,paramcache)
  b
end

function Algebra.allocate_jacobian(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @abstractmethod
end

function ODEs.jacobian_add!(
  A,
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A,
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,nlop,r,us,ws,paramcache)
  A
end

function Algebra.jacobian(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}})

  paramcache = allocate_paramcache(nlop,r;evaluated=true)
  jacobian(nlop,r,us,ws,paramcache)
end

function Algebra.jacobian(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  A = allocate_jacobian(nlop,r,us,paramcache)
  jacobian!(A,nlop,r,us,ws,paramcache)
  A
end

# compute space-time residuals/jacobians (no time marching)

function Algebra.residual(
  solver::ThetaMethod,
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  u::AbstractParamVector,
  u0::AbstractParamVector)

  u = state[1]
  dt,θ = solver.dt,solver.θ
  x = copy(u)
  uθ = copy(u)

  shift!(r,dt*(θ-1))
  shift!(uθ,u0,θ,1-θ)
  shift!(x,u0,1/dt,-1/dt)
  us = (uθ,x)
  b = residual(nlop,r,us)
  shift!(r,dt*(1-θ))

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  u::AbstractParamVector,
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

  A = jacobian(nlop,r,us,ws)
  shift!(r,dt*(1-θ))

  return A
end

function Algebra.residual(
  solver::ThetaMethod,
  nlop::NonlinearParamOperator{LinearParamODE},
  r::TransientRealization,
  u::AbstractParamVector,
  u0::AbstractParamVector)

  dt,θ = solver.dt,solver.θ
  x = copy(u)
  fill!(x,zero(eltype(x)))
  us = (x,x)

  shift!(r,dt*(θ-1))
  b = residual(nlop,r,us)
  shift!(r,dt*(1-θ))

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  nlop::NonlinearParamOperator{LinearParamODE},
  r::TransientRealization,
  u::AbstractParamVector,
  u0::AbstractParamVector)

  dt,θ = solver.dt,solver.θ
  ws = (1,1)
  x = copy(u)
  fill!(x,zero(eltype(x)))
  us = (x,x)

  shift!(r,dt*(θ-1))
  A = jacobian(nlop,r,us,ws)
  shift!(r,dt*(1-θ))

  return A
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
