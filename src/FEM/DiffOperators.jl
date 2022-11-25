function Gridap.ODEs.ODETools.time_derivative(f::Function)
  function time_derivative_f(x,t,μ)
    fxt = zero(return_type(f,x,t,μ))#zero(VectorValue)
    _time_derivative_f(f,x,t,μ,fxt)
  end
  time_derivative_f(x::VectorValue) = t -> time_derivative_f(x,t,μ)
  time_derivative_f(t,μ) = x -> time_derivative_f(x,t,μ)
end

function _time_derivative_f(f,x,t,μ,::Any)
  ForwardDiff.derivative(t->f(t,μ)(x),t)
end

function _time_derivative_f(f,x,t,μ,::VectorValue)
  VectorValue(ForwardDiff.derivative(t->get_array(f(t,μ)(x)),t))
end

function _time_derivative_f(f,x,t,μ,::TensorValue)
  TensorValue(ForwardDiff.derivative(t->get_array(f(t,μ)(x)),t))
end
