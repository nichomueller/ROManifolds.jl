# interface to deal with the inf-sup stability condition of saddle point problems

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},induced_norm::Function,tpspace,trial,test,coupling::Function)

  op = TransientParamFEOperator(res,jacs,induced_norm,tpspace,trial,test)
  TransientParamSaddlePointFEOp(op,coupling)
end

function TransientParamFEOperator(
  res::Function,jac::Function,induced_norm::Function,tpspace,trial,test,coupling::Function)

  op = TransientParamFEOperator(res,jac,induced_norm,tpspace,trial,test)
  TransientParamSaddlePointFEOp(op,coupling)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,induced_norm::Function,tpspace,trial,test,coupling::Function)

  op = TransientParamFEOperator(res,jac,jac_t,induced_norm,tpspace,trial,test)
  TransientParamSaddlePointFEOp(op,coupling)
end

function TransientParamFEOperator(
  res::Function,induced_norm::Function,tpspace,trial,test,coupling::Function;order::Integer=1)

  op = TransientParamFEOperator(res,induced_norm,tpspace,trial,test;order)
  TransientParamSaddlePointFEOp(op,coupling)
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,jacs::Tuple{Vararg{Function}},
  induced_norm::Function,tpspace,trial,test,coupling::Function;kwargs...)

  op = TransientParamLinearFEOperator(forms,res,jacs,induced_norm,tpspace,trial,test;kwargs...)
  TransientParamSaddlePointFEOp(op,coupling)
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,jac::Function,induced_norm::Function,
  tpspace,trial,test,coupling::Function;kwargs...)

  op = TransientParamLinearFEOperator(forms,res,jac,induced_norm,tpspace,trial,test;kwargs...)
  TransientParamSaddlePointFEOp(op,coupling)
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,jac::Function,jac_t::Function,
  induced_norm::Function,tpspace,trial,test,coupling::Function;kwargs...)

  op = TransientParamLinearFEOperator(forms,res,jac,jac_t,induced_norm,tpspace,trial,test;kwargs...)
  TransientParamSaddlePointFEOp(op,coupling)
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,induced_norm::Function,
  tpspace,trial,test,coupling::Function;order::Integer=1,kwargs...)

  op = TransientParamLinearFEOperator(forms,res,induced_norm,tpspace,trial,test;order,kwargs...)
  TransientParamSaddlePointFEOp(op,coupling)
end

struct TransientParamSaddlePointFEOp{T<:ODEParamOperatorType} <: TransientParamFEOperator{T}
  op::TransientParamFEOperator{T}
  coupling::Function
end

FESpaces.get_test(op::TransientParamSaddlePointFEOp) = get_test(op)
FESpaces.get_trial(op::TransientParamSaddlePointFEOp) = get_trial(op)
Polynomials.get_order(op::TransientParamSaddlePointFEOp) = get_order(op)
ODEs.get_res(op::TransientParamSaddlePointFEOp) = get_res(op)
ODEs.get_jacs(op::TransientParamSaddlePointFEOp) = get_jacs(op)
ODEs.get_forms(op::TransientParamSaddlePointFEOp) = get_forms(op)
ODEs.get_assembler(op::TransientParamSaddlePointFEOp) = get_assembler(op)
realization(op::TransientParamSaddlePointFEOp;kwargs...) = realization(op;kwargs...)
get_induced_norm(op::TransientParamSaddlePointFEOp) = get_induced_norm(op)
get_coupling(op::TransientParamSaddlePointFEOp) = op.coupling

function assemble_norm_matrix(op::TransientParamSaddlePointFEOp)
  assemble_norm_matrix(op.op)
end

function ODEs.get_assembler(op::TransientParamSaddlePointFEOp,r::TransientParamRealization)
  get_assembler(op.op,r)
end

function assemble_coupling_matrix(op::TransientParamSaddlePointFEOp)
  test = get_test(op)
  trial = evaluate(get_trial(op),nothing)
  c = get_coupling(op)
  assemble_matrix(c,trial,test)
end
