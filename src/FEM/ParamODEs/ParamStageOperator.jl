struct ParamStageOperator <: ParamNonlinearOperator
  op::ODEParamOperator
  cache::AbstractParamCache
  r::TransientRealization
  us::Function
  ws::Tuple{Vararg{Real}}
end

function Algebra.allocate_residual(nlop::NonlinearParamStageOperator,x::AbstractVector)
  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  allocate_residual(op,r,usx,cache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::NonlinearParamStageOperator,
  x::AbstractVector)

  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  residual!(b,op,r,usx,cache)
end

function Algebra.allocate_jacobian(nlop::NonlinearParamStageOperator,x::AbstractVector)
  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  allocate_jacobian(op,r,usx,cache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::NonlinearParamStageOperator,
  x::AbstractVector)

  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  ws = nlop.ws
  jacobian!(A,op,r,usx,ws,cache)
  A
end
