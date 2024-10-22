struct TransientRBNewtonOp <: RBNewtonOperator
  nlop::TransientRBOperator{NonlinearParamODE}
  lop::LinearParamStageOperator
  odeopcache
  rx::TransientRealization
  usx::Function
  ws::Tuple{Vararg{Real}}
  cache
end

function RBSteady.RBNewtonOperator(
  nlop::TransientRBOperator{<:NonlinearParamODE},
  lop::LinearParamStageOperator,
  odeopcache,
  rx::TransientRealization,
  usx::Function,
  ws::Tuple{Vararg{Real}},
  cache)

  TransientRBNewtonOp(nlop,lop,odeopcache,rx,usx,ws,cache)
end

RBSteady.get_linear_resjac(op::TransientRBNewtonOp) = (op.lop.A,op.lop.b)

function Algebra.allocate_residual(
  op::TransientRBNewtonOp,
  x::AbstractVector)

  nlop,odeopcache = op.nlop,op.odeopcache
  rx = op.rx
  usx = op.usx(x)
  allocate_residual(nlop,rx,usx,odeopcache)
end

function Algebra.residual!(
  nlb::Tuple,
  op::TransientRBNewtonOp,
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
  op::TransientRBNewtonOp,
  x::AbstractVector)

  nlop,odeopcache = op.nlop,op.odeopcache
  rx = op.rx
  usx = op.usx(x)
  allocate_jacobian(nlop,rx,usx,odeopcache)
end

function Algebra.jacobian!(
  nlA::Tuple,
  op::TransientRBNewtonOp,
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
