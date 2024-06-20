"""
    GenericParamLinearNonlinearFEOperator <: ParamFEOperator{LinearNonlinearParamEq}

Interface to accommodate the separation of terms depending on their linearity in
a nonlinear problem. This allows to build and store once and for all linear
residuals/jacobians, and in the Newton-like iterations only evaluate and assemble
only the nonlinear components

"""
struct GenericParamLinearNonlinearFEOperator <: ParamFEOperator{LinearNonlinearParamEq}
  op_linear::ParamFEOperator{LinearParamEq}
  op_nonlinear::ParamFEOperator{NonlinearParamEq}
end

"""
    ParamLinearNonlinearFEOperatorWithTrian <: ParamFEOperatorWithTrian{LinearNonlinearParamEq}

Is to a ParamFEOperatorWithTrian as a GenericParamLinearNonlinearFEOperator is to
a ParamFEOperator

"""
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

"""
    const ParamLinearNonlinearFEOperator = Union{
      GenericParamLinearNonlinearFEOperator,
      ParamLinearNonlinearFEOperatorWithTrian
    }

"""
const ParamLinearNonlinearFEOperator = Union{
  GenericParamLinearNonlinearFEOperator,
  ParamLinearNonlinearFEOperatorWithTrian
}

"""
    get_linear_operator(op::ParamFEOperator) -> ParamFEOperator

Returns the linear part of the operator of a linear-nonlinear FE operator; throws
an error if the input is not defined as a linear-nonlinear FE operator

"""
get_linear_operator(op::ParamLinearNonlinearFEOperator) = op.op_linear

"""
    get_nonlinear_operator(op::ParamFEOperator) -> ParamFEOperator

Returns the nonlinear part of the operator of a linear-nonlinear FE operator; throws
an error if the input is not defined as a linear-nonlinear FE operator

"""
get_nonlinear_operator(op::ParamLinearNonlinearFEOperator) = op.op_nonlinear

function FESpaces.get_test(op::ParamLinearNonlinearFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_linear)
end

function FESpaces.get_trial(op::ParamLinearNonlinearFEOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_linear)
end

function IndexMaps.get_index_map(op::ParamLinearNonlinearFEOperator)
  @check all(get_vector_index_map(op.op_linear) .== get_vector_index_map(op.op_nonlinear))
  @check all(get_matrix_index_map(op.op_linear) .== get_matrix_index_map(op.op_nonlinear))
  get_index_map(op.op_linear)
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

"""
    join_operators(op::ParamLinearNonlinearFEOperator) -> ParamFEOperator

Joins the linear/nonlinear parts of the operator and returns the result

"""
function join_operators(op::ParamLinearNonlinearFEOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  join_operators(op_lin,op_nlin)
end
