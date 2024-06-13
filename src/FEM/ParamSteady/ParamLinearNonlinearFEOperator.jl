# interface to accommodate the separation of terms depending on the linearity/nonlinearity

struct GenericParamLinearNonlinearFEOperator <: ParamFEOperator{LinearNonlinearParamEq}
  op_linear::ParamFEOperator{LinearParamEq}
  op_nonlinear::ParamFEOperator{NonlinearParamEq}
end

struct ParamLinearNonlinearFEOperatorWithTrian <: ParamFEOperatorWithTrian{LinearNonlinearParamEq}
  op_linear::ParamFEOperatorWithTrian{LinearParamEq}
  op_nonlinear::ParamFEOperatorWithTrian{NonlinearParamEq}
end

function ParamLinNonlinFEOperator(
  op_linear::ParamFEOperator,
  op_nonlinear::ParamFEOperator)
  GenericParamLinearNonlinearFEOperator(op_linear,op_nonlinear)
end

function ParamLinNonlinFEOperator(
  op_linear::ParamFEOperatorWithTrian,
  op_nonlinear::ParamFEOperatorWithTrian)
  ParamLinearNonlinearFEOperatorWithTrian(op_linear,op_nonlinear)
end

const ParamLinearNonlinearFEOperator = Union{
  GenericParamLinearNonlinearFEOperator,
  ParamLinearNonlinearFEOperatorWithTrian
}

get_linear_operator(op) = @abstractmethod
get_linear_operator(op::ParamLinearNonlinearFEOperator) = op.op_linear
get_nonlinear_operator(op) = @abstractmethod
get_nonlinear_operator(op::ParamLinearNonlinearFEOperator) = op.op_nonlinear

function FESpaces.get_test(op::ParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_linear)
end

function FESpaces.get_trial(op::ParamLinearNonlinearFEOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_linear)
end

function ParamDataStructures.realization(op::ParamLinearNonlinearFEOperator;kwargs...)
  realization(op.op_linear;kwargs...)
end

function assemble_norm_matrix(op::ParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  assemble_norm_matrix(op.op_linear)
end

function assemble_coupling_matrix(op::ParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  assemble_coupling_matrix(op.op_linear)
end

function join_operators(
  op_lin::ParamFEOperator,
  op_nlin::ParamFEOperator)

  @check get_trial(op_lin) == get_trial(op_nlin)
  @check get_test(op_lin) == get_test(op_nlin)
  @check op_lin.pspace === op_nlin.pspace

  trial = get_trial(op_lin)
  test = get_test(op_lin)

  res(μ,u,v) = op_lin.res(μ,u,v) + op_nlin.res(μ,u,v)
  jac(μ,u,du,v) = op_lin.jac(μ,u,du,v) + op_nlin.jac(μ,u,du,v)
  ParamFEOperator(res,jac,op_lin.induced_norm,op_lin.pspace,trial,test)
end

function join_operators(
  op_lin::ParamSaddlePointFEOp,
  op_nlin::ParamFEOperator)

  jop = join_operators(op_lin.op,op_nlin)
  ParamSaddlePointFEOp(jop,op_lin.coupling)
end

function join_operators(
  op_lin::ParamFEOperatorWithTrian,
  op_nlin::ParamFEOperatorWithTrian)

  set_op_lin = set_triangulation(op_lin)
  set_op_nlin = set_triangulation(op_nlin)
  join_operators(set_op_lin,set_op_nlin)
end

function join_operators(op::ParamLinearNonlinearFEOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  join_operators(op_lin,op_nlin)
end
