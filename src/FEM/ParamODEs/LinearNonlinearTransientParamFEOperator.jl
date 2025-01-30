"""
    struct LinearNonlinearTransientParamFEOperator{T} <: TransientParamFEOperator{LinearNonlinearParamODE,T}
      op_linear::TransientParamFEOperator{LinearParamODE,T}
      op_nonlinear::TransientParamFEOperator{NonlinearParamODE,T}
    end

Interface to accommodate the separation of terms depending on their linearity in
a nonlinear problem. This allows to build and store once and for all linear
residuals/jacobians, and in the Newton-like iterations only evaluate and assemble
only the nonlinear components
"""
struct LinearNonlinearTransientParamFEOperator{T} <: TransientParamFEOperator{LinearNonlinearParamODE,T}
  op_linear::TransientParamFEOperator{LinearParamODE,T}
  op_nonlinear::TransientParamFEOperator{NonlinearParamODE,T}
end

ParamSteady.get_linear_operator(op::LinearNonlinearTransientParamFEOperator) = op.op_linear
ParamSteady.get_nonlinear_operator(op::LinearNonlinearTransientParamFEOperator) = op.op_nonlinear

function FESpaces.get_algebraic_operator(feop::LinearNonlinearTransientParamFEOperator)
  LinearNonlinearParamOpFromTFEOp(feop)
end

function FESpaces.get_test(op::LinearNonlinearTransientParamFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_linear)
end

function FESpaces.get_trial(op::LinearNonlinearTransientParamFEOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_linear)
end

function ParamSteady.get_param_space(op::LinearNonlinearTransientParamFEOperator)
  @check get_param_space(op.op_linear) === get_param_space(op.op_nonlinear)
  get_param_space(op.op_linear)
end

function ReferenceFEs.get_order(op::LinearNonlinearTransientParamFEOperator)
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
  op_lin::TransientParamFEOperator,
  op_nlin::TransientParamFEOperator)

  op_lin = set_domains(op_lin)
  op_nlin = set_domains(op_nlin)

  @check get_trial(op_lin) == get_trial(op_nlin)
  @check get_test(op_lin) == get_test(op_nlin)
  @check op_lin.tpspace === op_nlin.tpspace

  trial = get_trial(op_lin)
  test = get_test(op_lin)
  order = max(get_order(op_lin),get_order(op_nlin))

  res(μ,t,u,v) = get_res(op_lin)(μ,t,u,v) + get_res(op_nlin)(μ,t,u,v)

  order_lin = get_order(op_lin)
  order_nlin = get_order(op_nlin)

  jacs = ()
  for i = 1:order+1
    function jac_i(μ,t,u,du,v)
      if i <= order_lin+1 && i <= order_nlin+1
        get_jacs(op_lin)[i](μ,t,u,du,v) + get_jacs(op_nlin)[i](μ,t,u,du,v)
      elseif i <= order_lin+1
        get_jacs(op_lin)[i](μ,t,u,du,v)
      else i <= order_nlin+1
        get_jacs(op_nlin)[i](μ,t,u,du,v)
      end
    end
    jacs = (jacs...,jac_i)
  end

  TransientParamFEOperator(res,jacs,op_lin.tpspace,trial,test)
end

for f in (:(ParamSteady.set_domains),:(ParamSteady.change_domains))
  @eval begin
    function $f(op::LinearNonlinearTransientParamFEOperator)
      op_lin′ = $f(get_linear_operator(op))
      op_nlin′ = $f(get_nonlinear_operator(op))
      LinearNonlinearTransientParamFEOperator(op_lin′,op_nlin′)
    end
  end
end

function ParamSteady.join_operators(op::LinearNonlinearTransientParamFEOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  join_operators(op_lin,op_nlin)
end
