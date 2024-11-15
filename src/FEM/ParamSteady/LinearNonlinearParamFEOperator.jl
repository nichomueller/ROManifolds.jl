"""
    struct LinearNonlinearParamFEOperator <: ParamFEOperator{LinearNonlinearParamEq} end

Interface to accommodate the separation of terms depending on their linearity in
a nonlinear problem. This allows to build and store once and for all linear
residuals/jacobians, and in the Newton-like iterations only evaluate and assemble
only the nonlinear components

"""
struct LinearNonlinearParamFEOperator{T} <: ParamFEOperator{LinearNonlinearParamEq,T}
  op_linear::ParamFEOperator{LinearParamEq,T}
  op_nonlinear::ParamFEOperator{NonlinearParamEq,T}
end

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

function FESpaces.get_algebraic_operator(op::LinearNonlinearParamFEOperator)
  LinearNonlinearParamOpFromFEOp(op)
end

function FESpaces.get_test(op::LinearNonlinearParamFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_linear)
end

function FESpaces.get_trial(op::LinearNonlinearParamFEOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_linear)
end

function get_fe_dof_map(op::LinearNonlinearParamFEOperator)
  @check all(get_dof_map(op.op_linear) .== get_dof_map(op.op_nonlinear))
  @check all(get_sparse_dof_map(op.op_linear) .== get_sparse_dof_map(op.op_nonlinear))
  get_fe_dof_map(op.op_linear)
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

  op_lin = set_domains(op_lin)
  op_nlin = set_domains(op_nlin)

  @check get_trial(op_lin) == get_trial(op_nlin)
  @check get_test(op_lin) == get_test(op_nlin)
  @check op_lin.pspace === op_nlin.pspace

  trial = get_trial(op_lin)
  test = get_test(op_lin)

  res(μ,u,v) = op_lin.res(μ,u,v) + op_nlin.res(μ,u,v)
  jac(μ,u,du,v) = op_lin.jac(μ,u,du,v) + op_nlin.jac(μ,u,du,v)
  ParamFEOperator(res,jac,op_lin.pspace,trial,test)
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
