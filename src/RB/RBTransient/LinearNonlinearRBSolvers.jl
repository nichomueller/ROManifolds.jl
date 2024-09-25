abstract type LinearNonlinearRBOperator <: NonlinearOperator end

struct LinearNonlinearRBSolver <: LinearNonlinearRBOperator
  nlop::TransientRBOperator{NonlinearParamODE}
  lop::LinearParamStageOperator
  odeopcache
  rx::TransientRealization
  usx::Function
  ws::Tuple{Vararg{Real}}
  cache
end

function LinearNonlinearRBSolver(
  op::TransientRBOperator{<:LinearNonlinearParamODE},
  lop::LinearParamStageOperator,
  odeopcache,
  rx::TransientRealization,
  usx::Function,
  ws::Tuple{Vararg{Real}},
  cache)

  nlop = get_nonlinear_operator(op)
  LinearNonlinearRBSolver(nlop,lop,odeopcache,rx,usx,ws,cache)
end

function Algebra.allocate_residual(
  op::LinearNonlinearRBSolver,
  x::AbstractVector)

  nlop,odeopcache = op.nlop,op.odeopcache
  rx = op.rx
  usx = op.usx(x)
  allocate_residual(nlop,rx,usx,odeopcache)
end

function Algebra.residual!(
  nlb::Tuple,
  op::LinearNonlinearRBSolver,
  x::AbstractVector)

  b̂lin = op.lop.b

  nlop,odeopcache = op.nlop,op.odeopcache
  rx = op.rx
  usx = op.usx(x)
  b̂nlin = residual!(nlb,nlop,rx,usx,odeopcache)

  @. b̂nlin = b̂nlin + b̂lin
  return b̂nlin
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearRBSolver,
  x::AbstractVector)

  nlop,odeopcache = op.nlop,op.odeopcache
  rx = op.rx
  usx = op.usx(x)
  allocate_jacobian(nlop,rx,usx,odeopcache)
end

function Algebra.jacobian!(
  nlA::Tuple,
  op::LinearNonlinearRBSolver,
  x::AbstractVector)

  Âlin = op.lop.A

  nlop,odeopcache = op.nlop,op.odeopcache
  rx = op.rx
  usx = op.usx(x)
  ws = op.ws
  Ânlin = jacobian!(nlA,nlop,rx,usx,ws,odeopcache)

  @. Ânlin = Ânlin + Âlin
  return Ânlin
end

function Algebra.solve!(
  x̂,
  nls::NewtonRaphsonSolver,
  op::LinearNonlinearRBSolver,
  r,
  x;
  verbose=true)

  Â_lin = op.lop.A
  syscache,trial = op.cache
  Â_cache,b̂_cache = syscache

  dx̂ = similar(x̂)
  Â = jacobian!(Â_cache,op,x)
  b̂ = residual!(b̂_cache,op,x)
  b̂ .+= Â_lin*x̂

  ss = symbolic_setup(nls.ls,Â)
  ns = numerical_setup(ss,Â)

  max0 = maximum(abs,b̂)
  tol = 1e-6*max0

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
