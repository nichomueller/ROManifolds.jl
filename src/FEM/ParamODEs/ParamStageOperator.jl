"""
    struct ParamStageOperator{O} <: StageOperator
      op::ODEParamOperator{O}
      cache::AbstractParamCache
      r::TransientRealization
      us::Function
      ws::Tuple{Vararg{Real}}
    end

(Nonlinear) stage operator to solve a parametric ODE with a time marching scheme
"""
struct ParamStageOperator{O} <: StageOperator
  op::ODEParamOperator{O}
  cache::AbstractParamCache
  r::TransientRealization
  us::Function
  ws::Tuple{Vararg{Real}}
end

function Algebra.allocate_residual(nlop::ParamStageOperator,x::AbstractParamVector)
  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  allocate_residual(op,r,usx,cache)
end

function Algebra.residual!(
  b::AbstractParamVector,
  nlop::ParamStageOperator,
  x::AbstractParamVector)

  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  residual!(b,op,r,usx,cache)
end

function Algebra.allocate_jacobian(nlop::ParamStageOperator,x::AbstractParamVector)
  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  allocate_jacobian(op,r,usx,cache)
end

function Algebra.jacobian!(
  A::AbstractParamMatrix,
  nlop::ParamStageOperator,
  x::AbstractParamVector)

  op = nlop.op
  cache = nlop.cache
  r = nlop.r
  usx = nlop.us(x)
  ws = nlop.ws
  jacobian!(A,op,r,usx,ws,cache)
  A
end

# linear case

function Algebra.allocate_residual(
  lop::ParamStageOperator{LinearParamODE},
  x::AbstractParamVector)

  cache = lop.cache
  b = copy(cache.b)
  fill!(b,zero(eltype(b)))
  b
end

function Algebra.residual!(
  b::AbstractParamVector,
  lop::ParamStageOperator{LinearParamODE},
  x::AbstractParamVector)

  op = lop.op
  cache = lop.cache.paramcache
  r = lop.r
  usx = lop.us(x)
  residual!(b,op,r,usx,cache)
end

function Algebra.allocate_jacobian(
  lop::ParamStageOperator{LinearParamODE},
  x::AbstractParamVector)

  cache = lop.cache
  A = copy(first(cache.A))
  A
end

function Algebra.jacobian!(
  A::AbstractParamMatrix,
  lop::ParamStageOperator{LinearParamODE},
  x::AbstractParamVector)

  op = lop.op
  cache = lop.cache
  r = lop.r
  usx = lop.us(x)
  ws = lop.ws
  jacobian!(A,op,r,usx,ws,cache)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  lop::ParamStageOperator{LinearParamODE},
  cache::Nothing)

  fill!(x,zero(eltype(x)))
  b = residual(lop,x)
  rmul!(b,-1)
  A = jacobian(lop,x)
  ns = solve!(x,ls,A,b)

  Algebra.LinearSolverCache(A,b,ns)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  lop::ParamStageOperator{LinearParamODE},
  cache::Algebra.LinearSolverCache)

  fill!(x,zero(eltype(x)))
  b = cache.b
  A = cache.A
  ns = cache.ns
  residual!(b,lop,x)
  rmul!(b,-1)
  jacobian!(A,lop,x)
  ns = solve!(x,ls,A,b)
  cache
end
