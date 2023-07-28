function time_derivative(f::Function)
  function time_derivative_f(x,μ,t)
    fxt = zero(return_type(f,x,μ,t))
    _time_derivative_f(f,x,μ,t,fxt)
  end
  time_derivative_f(x::VectorValue) = t -> time_derivative_f(x,μ,t)
  time_derivative_f(μ,t) = x -> time_derivative_f(x,μ,t)
end

function _time_derivative_f(f,x,μ,t,::Any)
  ForwardDiff.derivative(t->f(μ,t)(x),t)
end

function _time_derivative_f(f,x,μ,t,::VectorValue)
  VectorValue(ForwardDiff.derivative(t->get_array(f(μ,t)(x)),t))
end

function _time_derivative_f(f,x,μ,t,::TensorValue)
  TensorValue(ForwardDiff.derivative(t->get_array(f(μ,t)(x)),t))
end
