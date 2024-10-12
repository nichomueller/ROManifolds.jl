abstract type RBNewtonRaphsonOperator <: NonlinearOperator end

function Algebra.solve!(
  x̂,
  nls,
  op::RBNewtonRaphsonOperator,
  r,
  x;
  verbose=true)

  Â_lin,b̂_lin = get_linear_resjac(op)
  syscache,trial = op.cache
  Â_cache,b̂_cache = syscache

  dx̂ = similar(x̂)
  Â = jacobian!(Â_cache,op,x)
  b̂ = residual!(b̂_cache,op,x)
  b̂ .+= Â_lin*x̂

  ss = symbolic_setup(nls.ls,Â)
  ns = numerical_setup(ss,Â)

  max0 = maximum(abs,b̂)
  tol = max(1e-6*max0,eps())

  for k in 1:nls.max_nliters
    rmul!(b̂,-1)
    solve!(dx̂,ns,b̂)
    x̂ .+= dx̂
    inv_project!(x,trial,x̂)

    b̂ = residual!(b̂_cache,op,x)
    Â = jacobian!(Â_cache,op,x)
    numerical_setup!(ns,Â)

    b̂ .+= Â_lin*x̂
    maxk = maximum(abs,b̂)
    if verbose
      println("Newton-Raphson residual in the L∞ norm at iteration $(k) is $(maxk)")
    end

    maxk < tol && return

    if k == nls.max_nliters
      @unreachable "Newton-Raphson failed to converge: did not reach tolerance $tol"
    end
  end
end

get_linear_resjac(op::RBNewtonRaphsonOperator) = @abstractmethod

struct RBNewtonRaphsonOp <: RBNewtonRaphsonOperator
  op::ParamOperator{NonlinearParamEq}
  paramcache::ParamCache
  r::Realization
  Â_lin::AbstractMatrix
  b̂_lin::AbstractVector
  cache
end

function RBNewtonRaphsonOperator(
  op::ParamOperator{NonlinearParamEq},
  paramcache::ParamCache,
  r::Realization,
  Â_lin::AbstractMatrix,
  b̂_lin::AbstractVector,
  cache)

  RBNewtonRaphsonOp(op,paramcache,r,Â_lin,b̂_lin,cache)
end

get_linear_resjac(op::RBNewtonRaphsonOp) = (op.Â_lin,op.b̂_lin)

function Algebra.allocate_residual(
  op::RBNewtonRaphsonOp,
  u::AbstractVector)

  paramcache = op.paramcache
  paramcache.b
end

function Algebra.residual!(
  nlb::Tuple,
  op::RBNewtonRaphsonOp,
  x::AbstractVector)

  b̂lin = op.b̂_lin

  paramcache = op.paramcache
  b̂nlin = residual!(nlb,op.op,op.r,x,paramcache)

  @. b̂nlin = b̂nlin + b̂lin
  return b̂nlin
end

function Algebra.allocate_jacobian(
  op::RBNewtonRaphsonOp,
  x::AbstractVector)

  paramcache = op.paramcache
  paramcache.A
end

function Algebra.jacobian!(
  nlA::Tuple,
  op::RBNewtonRaphsonOp,
  x::AbstractVector)

  Âlin = op.Â_lin

  paramcache = op.paramcache
  Ânlin = jacobian!(nlA,op.op,op.r,x,paramcache)

  @. Ânlin = Ânlin + Âlin
  return Ânlin
end
