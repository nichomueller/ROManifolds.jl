function ODEs.time_derivative(f::TransientParamFunction)
  function dfdt(μ,t)
    fμt = f.fun(μ,t)
    function dfdt_t(x)
      T = return_type(fμt,x)
      ODEs._time_derivative(T,f.fun,μ,t,x)
    end
    dfdt_t
  end
  TransientParamFunction(dfdt,f.params,f.times)
end

function ODEs._time_derivative(T::Type{<:Real},f,μ,t,x)
  partial(t) = f(μ,t)(x)
  ForwardDiff.derivative(partial,t)
end

function ODEs._time_derivative(T::Type{<:VectorValue},f,μ,t,x)
  partial(t) = get_array(f(μ,t)(x))
  VectorValue(ForwardDiff.derivative(partial,t))
end

function ODEs._time_derivative(T::Type{<:TensorValue},f,μ,t,x)
  partial(t) = get_array(f(μ,t)(x))
  TensorValue(ForwardDiff.derivative(partial,t))
end
