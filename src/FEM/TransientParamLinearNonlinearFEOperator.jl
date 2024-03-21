# interface to accommodate the separation of terms depending on the linearity/nonlinearity
struct LinearNonlinearParamODE <: ODEParamOperatorType end

struct TransientParamLinearNonlinearFEOperator <: TransientParamFEOperator{LinearNonlinearParamODE}
  op_linear::TransientParamFEOperator{LinearParamODE}
  op_nonlinear::TransientParamFEOperator
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

function Polynomials.get_order(op::TransientParamLinearNonlinearFEOperator)
  return max(get_order(op.op_linear),get_order(op.op_nonlinear))
end

function realization(op::TransientParamLinearNonlinearFEOperator;kwargs...)
  realization(op.op_linear;kwargs...)
end

function assemble_norm_matrix(op::TransientParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  assemble_norm_matrix(op.op_linear)
end

function assemble_coupling_matrix(op::TransientParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  assemble_coupling_matrix(op.op_linear)
end

function _get_res(op::TransientParamFEOperator)
  return op.res
end

function _get_res(op::TransientParamFEOperator{SemilinearParamODE})
  mass = get_forms(op)[1]
  function res_semilin(μ,t,u,v)
    ∂tu = ∂t(u,Val(op.order))
    res = op.res(μ,t,u,v) + mass(μ,t,∂tu,v)
    return res
  end
  return res_semilin
end

function _get_res(op::TransientParamFEOperator{LinearParamODE})
  forms = get_forms(op)
  function res_lin(μ,t,u,v)
    ∂tu = u
    res = op.res(μ,t,u,v)
    for k = 1:op.order+1
      res += forms(μ,t,∂tu,v)
      if k <= op.order
        ∂tu = ∂t(∂tu)
      end
    end
    return res
  end
  return res_lin
end

function _join_res(op_lin,op_nlin)
  (μ,t,u,v) -> _get_res(op_lin)(μ,t,u,v) + _get_res(op_nlin)(μ,t,u,v)
end

function join_operators(
  op::TransientParamLinearNonlinearFEOperator,
  op_lin::TransientParamFEOperator{LinearParamODE},
  op_nlin::TransientParamFEOperator)

  trial = get_trial(op)
  test = get_test(op)
  @check op_lin.tpspace === op_nlin.tpspace

  res = _join_res(op_lin,op_nlin)

  order_lin = get_order(op_lin)
  order_nlin = get_order(op_nlin)

  jacs = ()
  for i = 1:get_order(op)+1
    function jac_i(μ,t,u,du,v)
      if i <= order_lin+1 && i <= order_nlin+1
        op_lin.jacs[i](μ,t,u,du,v) + op_nlin.jacs[i](μ,t,u,du,v)
      elseif i <= order_lin+1
        op_lin.jacs[i](μ,t,u,du,v)
      else i <= order_nlin+1
        op_nlin.jacs[i](μ,t,u,du,v)
      end
    end
    jacs = (jacs...,jac_i)
  end

  TransientParamFEOperator(res,jacs,op_lin.induced_norm,op_lin.tpspace,trial,test)
end

function join_operators(
  op::TransientParamLinearNonlinearFEOperator,
  op_lin::TransientParamSaddlePointFEOp{LinearParamODE},
  op_nlin::TransientParamFEOperator)

  jop = join_operators(op,op_lin.op,op_nlin)
  TransientParamSaddlePointFEOp(jop,op_lin.coupling)
end

function join_operators(
  op::TransientParamLinearNonlinearFEOperator,
  op_lin::TransientParamFEOperatorWithTrian,
  op_nlin::TransientParamFEOperatorWithTrian)

  set_op_lin = set_triangulation(op_lin)
  set_op_nlin = set_triangulation(op_nlin)
  join_operators(op,set_op_lin,set_op_nlin)
end

function join_operators(op::TransientParamLinearNonlinearFEOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  join_operators(op,op_lin,op_nlin)
end
