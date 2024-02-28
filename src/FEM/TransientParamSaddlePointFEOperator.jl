# interface to deal with the inf-sup stability condition of saddle point problems

function AffineTransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,tpspace,trial,test,coupling::Function)

  op = AffineTransientParamFEOperator(res,jac,induced_norm,tpspace,trial,test)
  TransientParamSaddlePointFEOperator(op,coupling)
end

function AffineTransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,tpspace,trial,test,coupling::Function)

  op = AffineTransientParamFEOperator(res,jac,jac_t,induced_norm,tpspace,trial,test)
  TransientParamSaddlePointFEOperator(op,coupling)
end

function TransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,tpspace,trial,test,coupling::Function)

  op = TransientParamFEOperator(res,jac,induced_norm,tpspace,trial,test)
  TransientParamSaddlePointFEOperator(op,coupling)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,tpspace,trial,test,coupling::Function)

  op = TransientParamFEOperator(res,jac,jac_t,induced_norm,tpspace,trial,test)
  TransientParamSaddlePointFEOperator(op,coupling)
end

struct TransientParamSaddlePointFEOperator{T<:OperatorType} <: TransientParamFEOperator{T}
  op::TransientParamFEOperator{T}
  coupling::Function
end

FESpaces.get_test(op::TransientParamSaddlePointFEOperator) = get_test(op.op)
FESpaces.get_trial(op::TransientParamSaddlePointFEOperator) = get_trial(op.op)
ReferenceFEs.get_order(op::TransientParamSaddlePointFEOperator) = get_order(op.op)
realization(op::TransientParamSaddlePointFEOperator;kwargs...) = realization(op.op;kwargs...)

function assemble_norm_matrix(op::TransientParamSaddlePointFEOperator)
  assemble_norm_matrix(op.op)
end

function assemble_coupling_matrix(op::TransientParamSaddlePointFEOperator)
  test = get_test(op)
  trial = evaluate(get_trial(op),(nothing))
  assemble_matrix(op.coupling,trial,test)
end

function Algebra.allocate_residual(
  op::TransientParamSaddlePointFEOperator,
  r::TransientParamRealization,
  uh::T,
  cache) where T

  allocate_residual(op.op,r,uh,cache)
end

function Algebra.allocate_jacobian(
  op::TransientParamSaddlePointFEOperator,
  r::TransientParamRealization,
  uh::CellField,
  cache)

  allocate_jacobian(op.op,r,uh,cache)
end

function Algebra.residual!(b,op::TransientParamSaddlePointFEOperator,args...)
  residual!(b,op.op,args...)
end

function Algebra.jacobian!(A,op::TransientParamSaddlePointFEOperator,args...)
  jacobian!(A,op.op,args...)
end

function ODETools.jacobians!(A,op::TransientParamSaddlePointFEOperator,args...)
  jacobians!(A,op.op,args...)
end
