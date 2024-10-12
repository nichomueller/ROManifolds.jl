struct TransientRBNewtonRaphsonOp <: RBNewtonRaphsonOperator
  nlop::TransientRBOperator{NonlinearParamODE}
  lop::LinearParamStageOperator
  odeopcache
  rx::TransientRealization
  usx::Function
  ws::Tuple{Vararg{Real}}
  cache
end

function RBSteady.RBNewtonRaphsonOperator(
  nlop::TransientRBOperator{<:NonlinearParamODE},
  lop::LinearParamStageOperator,
  odeopcache,
  rx::TransientRealization,
  usx::Function,
  ws::Tuple{Vararg{Real}},
  cache)

  TransientRBNewtonRaphsonOp(nlop,lop,odeopcache,rx,usx,ws,cache)
end

RBSteady.get_linear_resjac(op::TransientRBNewtonRaphsonOp) = (op.lop.A,op.lop.b)

function Algebra.allocate_residual(
  op::TransientRBNewtonRaphsonOp,
  x::AbstractVector)

  nlop,odeopcache = op.nlop,op.odeopcache
  rx = op.rx
  usx = op.usx(x)
  allocate_residual(nlop,rx,usx,odeopcache)
end

function Algebra.residual!(
  nlb::Tuple,
  op::TransientRBNewtonRaphsonOp,
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
  op::TransientRBNewtonRaphsonOp,
  x::AbstractVector)

  nlop,odeopcache = op.nlop,op.odeopcache
  rx = op.rx
  usx = op.usx(x)
  allocate_jacobian(nlop,rx,usx,odeopcache)
end

function Algebra.jacobian!(
  nlA::Tuple,
  op::TransientRBNewtonRaphsonOp,
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
