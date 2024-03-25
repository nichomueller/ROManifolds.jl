function ODEs.time_derivative(f::TransientParamFunction)
  @unpack fun,params,times = f
  function dfdt(x,μ,t)
    z = zero(return_type(fun,x,μ,t))
    ODEs._time_derivative(fun,x,μ,t,z)
  end
  _dfdt(x,μ,t) = dfdt(x,μ,t)
  _dfdt(x::VectorValue) = (μ,t) -> dfdt(x,μ,t)
  _dfdt(μ,t) = x -> dfdt(x,μ,t)
  _dfdt(x,r::TransientParamRealization) = dfdt(x,get_params(r),get_times(r))
  _dfdt(r::TransientParamRealization) = x -> dfdt(x,get_params(r),get_times(r))
  return TransientParamFunction(_dfdt,params,times)
end

function ODEs._time_derivative(f,x,μ,t,::Any)
  ForwardDiff.derivative(t->f(x,μ,t),t)
end

function ODEs._time_derivative(f,x,μ,t,::VectorValue)
  VectorValue(ForwardDiff.derivative(t->get_array(f(x,μ,t)),t))
end

function ODEs._time_derivative(f,x,μ,t,::TensorValue)
  TensorValue(ForwardDiff.derivative(t->get_array(f(x,μ,t)),t))
end
