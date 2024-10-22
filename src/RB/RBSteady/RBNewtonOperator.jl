abstract type RBNewtonOperator <: NonlinearOperator end

function Algebra.solve!(
  x̂,
  nls::NonlinearSolvers.NewtonSolver,
  op::RBNewtonOperator,
  r,
  x)

  Â_lin,b̂_lin = get_linear_resjac(op)
  syscache,trial = op.cache
  Â_cache,b̂_cache = syscache

  dx̂ = similar(x̂)
  Â = jacobian!(Â_cache,op,x)
  b̂ = residual!(b̂_cache,op,x)
  b̂ .+= Â_lin*x̂

  ss = symbolic_setup(nls.ls,Â)
  ns = numerical_setup(ss,Â)

  res = norm(b̂)
  done = LinearSolvers.init!(nls.log,res)

  while !done
    rmul!(b̂,-1)
    solve!(dx̂,ns,b̂)
    x̂ .+= dx̂

    b̂ = residual!(b̂_cache,op,x)
    b̂ .+= Â_lin*x̂
    res  = norm(b̂)
    done = LinearSolvers.update!(nls.log,res)

    if !done
      inv_project!(x,trial,x̂)
      Â = jacobian!(Â_cache,op,x)
      numerical_setup!(ns,Â)
    end
  end

  LinearSolvers.finalize!(nls.log,res)
  return x̂
end

get_linear_resjac(op::RBNewtonOperator) = @abstractmethod

struct RBNewtonOp <: RBNewtonOperator
  op::ParamOperator{NonlinearParamEq}
  paramcache::ParamCache
  r::Realization
  Â_lin::AbstractMatrix
  b̂_lin::AbstractVector
  cache
end

function RBNewtonOperator(
  op::ParamOperator{NonlinearParamEq},
  paramcache::ParamCache,
  r::Realization,
  Â_lin::AbstractMatrix,
  b̂_lin::AbstractVector,
  cache)

  RBNewtonOp(op,paramcache,r,Â_lin,b̂_lin,cache)
end

get_linear_resjac(op::RBNewtonOp) = (op.Â_lin,op.b̂_lin)

function Algebra.allocate_residual(
  op::RBNewtonOp,
  u::AbstractVector)

  paramcache = op.paramcache
  paramcache.b
end

function Algebra.residual!(
  nlb::Tuple,
  op::RBNewtonOp,
  x::AbstractVector)

  b̂lin = op.b̂_lin

  paramcache = op.paramcache
  b̂nlin = residual!(nlb,op.op,op.r,x,paramcache)

  @. b̂nlin = b̂nlin + b̂lin
  return b̂nlin
end

function Algebra.allocate_jacobian(
  op::RBNewtonOp,
  x::AbstractVector)

  paramcache = op.paramcache
  paramcache.A
end

function Algebra.jacobian!(
  nlA::Tuple,
  op::RBNewtonOp,
  x::AbstractVector)

  Âlin = op.Â_lin

  paramcache = op.paramcache
  Ânlin = jacobian!(nlA,op.op,op.r,x,paramcache)

  @. Ânlin = Ânlin + Âlin
  return Ânlin
end
