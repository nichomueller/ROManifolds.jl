function p_time_derivative(f::Function)
  function p_time_derivative_f(x,μ,t)
    fxt = zero(return_type(f,x,μ,t))
    _p_time_derivative_f(f,x,μ,t,fxt)
  end
  p_time_derivative_f(x::VectorValue) = t -> p_time_derivative_f(x,μ,t)
  p_time_derivative_f(μ,t) = x -> p_time_derivative_f(x,μ,t)
end

function _p_time_derivative_f(f,x,μ,t,::Any)
  ForwardDiff.derivative(t->f(μ,t)(x),t)
end

function _p_time_derivative_f(f,x,μ,t,::VectorValue)
  VectorValue(ForwardDiff.derivative(t->get_array(f(μ,t)(x)),t))
end

function _p_time_derivative_f(f,x,μ,t,::TensorValue)
  TensorValue(ForwardDiff.derivative(t->get_array(f(μ,t)(x)),t))
end

const ∂ₚt = p_time_derivative

#default
∂ₚt(f) = ∂t(f)
