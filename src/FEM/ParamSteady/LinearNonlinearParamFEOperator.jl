"""
    struct GenericLinearNonlinearParamFEOperator <: ParamFEOperator{LinearNonlinearParamEq} end

Interface to accommodate the separation of terms depending on their linearity in
a nonlinear problem. This allows to build and store once and for all linear
residuals/jacobians, and in the Newton-like iterations only evaluate and assemble
only the nonlinear components

"""
struct GenericLinearNonlinearParamFEOperator <: ParamFEOperator{LinearNonlinearParamEq}
  op_linear::ParamFEOperator{LinearParamEq}
  op_nonlinear::ParamFEOperator{NonlinearParamEq}
end

"""
    struct LinearNonlinearParamFEOperatorWithTrian <: ParamFEOperatorWithTrian{LinearNonlinearParamEq} end

Is to a ParamFEOperatorWithTrian as a GenericLinearNonlinearParamFEOperator is to
a ParamFEOperator

"""
struct LinearNonlinearParamFEOperatorWithTrian <: ParamFEOperatorWithTrian{LinearNonlinearParamEq}
  op_linear::ParamFEOperatorWithTrian{LinearParamEq}
  op_nonlinear::ParamFEOperatorWithTrian{NonlinearParamEq}
end

function LinNonlinParamFEOperator(
  op_linear::ParamFEOperator,
  op_nonlinear::ParamFEOperator)
  GenericLinearNonlinearParamFEOperator(op_linear,op_nonlinear)
end

function LinNonlinParamFEOperator(
  op_linear::ParamFEOperatorWithTrian,
  op_nonlinear::ParamFEOperatorWithTrian)
  LinearNonlinearParamFEOperatorWithTrian(op_linear,op_nonlinear)
end

"""
    const LinearNonlinearParamFEOperator = Union{
      GenericLinearNonlinearParamFEOperator,
      LinearNonlinearParamFEOperatorWithTrian
    }

"""
const LinearNonlinearParamFEOperator = Union{
  GenericLinearNonlinearParamFEOperator,
  LinearNonlinearParamFEOperatorWithTrian
}

"""
    get_linear_operator(op::ParamFEOperator) -> ParamFEOperator

Returns the linear part of the operator of a linear-nonlinear FE operator; throws
an error if the input is not defined as a linear-nonlinear FE operator

"""
get_linear_operator(op::LinearNonlinearParamFEOperator) = op.op_linear

"""
    get_nonlinear_operator(op::ParamFEOperator) -> ParamFEOperator

Returns the nonlinear part of the operator of a linear-nonlinear FE operator; throws
an error if the input is not defined as a linear-nonlinear FE operator

"""
get_nonlinear_operator(op::LinearNonlinearParamFEOperator) = op.op_nonlinear

function FESpaces.get_algebraic_operator(op::LinearNonlinearParamFEOperator,r::Realization)
  LinearNonlinearParamOpFromFEOp(op,r)
end

function FESpaces.get_test(op::LinearNonlinearParamFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_linear)
end

function FESpaces.get_trial(op::LinearNonlinearParamFEOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_linear)
end

function IndexMaps.get_index_map(op::LinearNonlinearParamFEOperator)
  @check all(get_vector_index_map(op.op_linear) .== get_vector_index_map(op.op_nonlinear))
  @check all(get_matrix_index_map(op.op_linear) .== get_matrix_index_map(op.op_nonlinear))
  get_index_map(op.op_linear)
end

function ParamDataStructures.realization(op::LinearNonlinearParamFEOperator;kwargs...)
  realization(op.op_linear;kwargs...)
end

function FESpaces.assemble_matrix(op::LinearNonlinearParamFEOperator,form::Function)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  assemble_matrix(op.op_linear,form)
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
  ParamFEOperator(res,jac,op_lin.pspace,trial,test)
end

function join_operators(
  op_lin::ParamFEOperatorWithTrian,
  op_nlin::ParamFEOperatorWithTrian)

  set_op_lin = set_triangulation(op_lin)
  set_op_nlin = set_triangulation(op_nlin)
  join_operators(set_op_lin,set_op_nlin)
end

"""
    join_operators(op::LinearNonlinearParamFEOperator) -> ParamFEOperator

Joins the linear/nonlinear parts of the operator and returns the result

"""
function join_operators(op::LinearNonlinearParamFEOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  join_operators(op_lin,op_nlin)
end
