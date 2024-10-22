"""
    struct LinearNonlinearTransientParamFEOperator <:
      TransientParamFEOperator{LinearNonlinearParamODE} end

Interface to accommodate the separation of terms depending on their linearity in
a nonlinear problem. This allows to build and store once and for all linear
residuals/jacobians, and in the Newton-like iterations only evaluate and assemble
only the nonlinear components

"""
struct LinearNonlinearTransientParamFEOperator <: TransientParamFEOperator{LinearNonlinearParamODE}
  op_linear::TransientParamFEOperator{LinearParamODE}
  op_nonlinear::TransientParamFEOperator
end

ParamSteady.get_linear_operator(op::LinearNonlinearTransientParamFEOperator) = op.op_linear
ParamSteady.get_nonlinear_operator(op::LinearNonlinearTransientParamFEOperator) = op.op_nonlinear

function FESpaces.get_algebraic_operator(feop::LinearNonlinearTransientParamFEOperator)
  ODEParamOpFromTFEOpWithTrian(feop)
end

function FESpaces.get_test(op::LinearNonlinearTransientParamFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_linear)
end

function FESpaces.get_trial(op::LinearNonlinearTransientParamFEOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_linear)
end

function IndexMaps.get_index_map(op::LinearNonlinearTransientParamFEOperator)
  @check all(get_vector_index_map(op.op_linear) .== get_vector_index_map(op.op_nonlinear))
  @check all(get_matrix_index_map(op.op_linear) .== get_matrix_index_map(op.op_nonlinear))
  get_index_map(op.op_linear)
end

function Polynomials.get_order(op::LinearNonlinearTransientParamFEOperator)
  return max(get_order(op.op_linear),get_order(op.op_nonlinear))
end

function ParamDataStructures.realization(op::LinearNonlinearTransientParamFEOperator;kwargs...)
  realization(op.op_linear;kwargs...)
end

function FESpaces.assemble_matrix(op::LinearNonlinearTransientParamFEOperator,form::Function)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  assemble_matrix(op.op_linear,form)
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

  TransientParamFEOperator(res,jacs,op_lin.tpspace,trial,test)
end

function ParamSteady.join_operators(
  op_lin::TransientParamFEOperatorWithTrian,
  op_nlin::TransientParamFEOperatorWithTrian)

  set_op_lin = set_triangulation(op_lin)
  set_op_nlin = set_triangulation(op_nlin)
  join_operators(set_op_lin,set_op_nlin)
end

function ParamSteady.join_operators(op::LinearNonlinearTransientParamFEOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  join_operators(op_lin,op_nlin)
end
