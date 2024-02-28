# interface to accommodate the separation of terms depending on the linearity/nonlinearity
struct LinearNonlinear <: OperatorType end

struct TransientParamLinearNonlinearFEOperator{A,B} <: TransientParamFEOperator{LinearNonlinear}
  op_linear::A
  op_nonlinear::B
  function TransientParamLinearNonlinearFEOperator(op_linear::A,op_nonlinear::B) where {A,B}
    @check isa(op_linear,TransientParamFEOperator{Affine})
    @check isa(op_nonlinear,TransientParamFEOperator{Nonlinear})
    new{A,B}(op_linear,op_nonlinear)
  end
end

get_linear_operator(op) = @abstractmethod
get_linear_operator(op::TransientParamLinearNonlinearFEOperator) = op.op_linear
get_nonlinear_operator(op) = @abstractmethod
get_nonlinear_operator(op::TransientParamLinearNonlinearFEOperator) = op.op_nonlinear

function FESpaces.get_test(op::TransientParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_linear)
end

function FESpaces.get_trial(op::TransientParamLinearNonlinearFEOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_linear)
end

function FESpaces.get_order(op::TransientParamLinearNonlinearFEOperator)
  return max(get_order(op.op_linear),get_order(op.op_nonlinear))
end

function realization(op::TransientParamLinearNonlinearFEOperator;kwargs...)
  realization(op.op_linear;kwargs...)
end

function assemble_norm_matrix(op::TransientParamLinearNonlinearFEOperator)
  test = get_test(op)
  trial = evaluate(get_trial(op),(nothing))
  assemble_matrix(op.op_linear.induced_norm,trial,test)
end

function compute_coupling_matrix(op::TransientParamLinearNonlinearFEOperator)
  test = get_test(op)
  trial = evaluate(get_trial(op),(nothing))
  assemble_matrix(op.op_linear.coupling,trial,test)
end

function join_operators(op::TransientParamLinearNonlinearFEOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  trial = get_trial(op)
  test = get_test(op)
  @check op_lin.op.tpspace === op_nlin.op.tpspace

  res(μ,t,u,v) = op_lin.res(μ,t,u,v) + op_nlin.res(μ,t,u,v)

  order_lin = get_order(newop_lin)
  order_nlin = get_order(newop_nlin)

  jacs = ()
  for i = 1:get_order(op)
    function jac_i(μ,t,u,du,v)
      if i > order_lin
        newop_nlin.op.jacs[i](μ,t,u,du,v)
      elseif i > order_nlin
        newop_lin.op.jacs[i](μ,t,u,du,v)
      else
        newop_lin.op.jacs[i](μ,t,u,du,v) + newop_nlin.op.jacs[i](μ,t,u,du,v)
      end
    end
    jacs = (jacs...,jac_i)
  end

  if isa(op_lin,TransientParamSaddlePointFEOperator)
    TransientParamFEOperator(res,jacs...,op_lin.op.induced_norm,op_lin.tpspace,trial,test,op_lin.op.coupling)
  else
    TransientParamFEOperator(res,jacs...,op_lin.op.induced_norm,op_lin.tpspace,trial,test)
  end
end

function join_operators(
  op::TransientParamLinearNonlinearFEOperator{A,B}
  ) where {A<:TransientParamFEOperatorWithTrian,B<:TransientParamFEOperatorWithTrian}

  join_operators(op.op)
end
