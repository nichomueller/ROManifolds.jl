"""
    GenericTransientParamLinearNonlinearFEOperator <:
      TransientParamFEOperator{LinearNonlinearParamODE}

Interface to accommodate the separation of terms depending on their linearity in
a nonlinear problem. This allows to build and store once and for all linear
residuals/jacobians, and in the Newton-like iterations only evaluate and assemble
only the nonlinear components

"""
struct GenericTransientParamLinearNonlinearFEOperator <: TransientParamFEOperator{LinearNonlinearParamODE}
  op_linear::TransientParamFEOperator{LinearParamODE}
  op_nonlinear::TransientParamFEOperator
end

"""
   TransientParamLinearNonlinearFEOperatorWithTrian <:
    TransientParamFEOperatorWithTrian{LinearNonlinearParamODE}

Is to a TransientParamFEOperatorWithTrian as a GenericTransientParamLinearNonlinearFEOperator is to
a TransientParamFEOperator

"""
struct TransientParamLinearNonlinearFEOperatorWithTrian <: TransientParamFEOperatorWithTrian{LinearNonlinearParamODE}
  op_linear::TransientParamFEOperatorWithTrian{LinearParamODE}
  op_nonlinear::TransientParamFEOperatorWithTrian
end

function TransientParamLinNonlinFEOperator(
  op_linear::TransientParamFEOperator,
  op_nonlinear::TransientParamFEOperator)
  GenericTransientParamLinearNonlinearFEOperator(op_linear,op_nonlinear)
end

function TransientParamLinNonlinFEOperator(
  op_linear::TransientParamFEOperatorWithTrian,
  op_nonlinear::TransientParamFEOperatorWithTrian)
  TransientParamLinearNonlinearFEOperatorWithTrian(op_linear,op_nonlinear)
end

"""
    const TransientParamLinearNonlinearFEOperator = Union{
      GenericTransientParamLinearNonlinearFEOperator,
      TransientParamLinearNonlinearFEOperatorWithTrian
    }

"""
const TransientParamLinearNonlinearFEOperator = Union{
  GenericTransientParamLinearNonlinearFEOperator,
  TransientParamLinearNonlinearFEOperatorWithTrian
}

ParamSteady.get_linear_operator(op::TransientParamLinearNonlinearFEOperator) = op.op_linear
ParamSteady.get_nonlinear_operator(op::TransientParamLinearNonlinearFEOperator) = op.op_nonlinear

function FESpaces.get_test(op::TransientParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_linear)
end

function FESpaces.get_trial(op::TransientParamLinearNonlinearFEOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_linear)
end

function IndexMaps.get_index_map(op::TransientParamLinearNonlinearFEOperator)
  @check all(get_vector_index_map(op.op_linear) .== get_vector_index_map(op.op_nonlinear))
  @check all(get_matrix_index_map(op.op_linear) .== get_matrix_index_map(op.op_nonlinear))
  get_index_map(op.op_linear)
end

function ReferenceFEs.get_order(op::TransientParamLinearNonlinearFEOperator)
  return max(get_order(op.op_linear),get_order(op.op_nonlinear))
end

function ParamDataStructures.realization(op::TransientParamLinearNonlinearFEOperator;kwargs...)
  realization(op.op_linear;kwargs...)
end

function ParamSteady.assemble_norm_matrix(op::TransientParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  assemble_norm_matrix(op.op_linear)
end

function ParamSteady.assemble_coupling_matrix(op::TransientParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  assemble_coupling_matrix(op.op_linear)
end

function ParamSteady.join_operators(
  op_lin::TransientParamFEOperator{LinearParamODE},
  op_nlin::TransientParamFEOperator)

  @check get_trial(op_lin) == get_trial(op_nlin)
  @check get_test(op_lin) == get_test(op_nlin)
  @check op_lin.tpspace === op_nlin.tpspace

  trial = get_trial(op_lin)
  test = get_test(op_lin)
  order = max(get_order(op_lin),get_order(op_nlin))

  res(μ,t,u,v) = op_lin.res(μ,t,u,v) + op_nlin.res(μ,t,u,v)

  order_lin = get_order(op_lin)
  order_nlin = get_order(op_nlin)

  jacs = ()
  for i = 1:order+1
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

function ParamSteady.join_operators(
  op_lin::TransientParamSaddlePointFEOp,
  op_nlin::TransientParamFEOperator)

  jop = join_operators(op_lin.op,op_nlin)
  TransientParamSaddlePointFEOp(jop,op_lin.coupling)
end

function ParamSteady.join_operators(
  op_lin::TransientParamFEOperatorWithTrian,
  op_nlin::TransientParamFEOperatorWithTrian)

  set_op_lin = set_triangulation(op_lin)
  set_op_nlin = set_triangulation(op_nlin)
  join_operators(set_op_lin,set_op_nlin)
end

function ParamSteady.join_operators(op::TransientParamLinearNonlinearFEOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  join_operators(op_lin,op_nlin)
end
