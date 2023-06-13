function Gridap.ODEs.ODETools.time_derivative(f::Function)
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

function param_derivative(f::Function)
  function param_derivative_f(x,μ)
    fxt = zero(return_type(f,x,μ))
    _param_derivative_f(f,x,μ,fxt)
  end
  param_derivative_f(x::VectorValue) = μ -> param_derivative_f(x,μ)
  param_derivative_f(μ) = x -> param_derivative_f(x,μ)
end

const ∂μ = param_derivative

function _param_derivative_f(f,x,μ,::Any)
  ForwardDiff.gradient(μ->f(x,μ),μ)
end

function _param_derivative_f(f,x,μ,::VectorValue)
  VectorValue(ForwardDiff.gradient(μ->get_array(f(x,μ)),μ))
end

function _param_derivative_f(f,x,μ,::TensorValue)
  TensorValue(ForwardDiff.gradient(μ->get_array(f(x,μ)),μ))
end

function param_time_derivative(f::Function)
  function param_time_derivative_f(x,μ,t)
    fxt = zero(return_type(f,x,μ,t))
    _param_time_derivative_f(f,x,μ,t,fxt)
  end
  param_time_derivative_f(x::VectorValue) = t -> param_time_derivative_f(x,μ,t)
  param_time_derivative_f(μ,t) = x -> param_time_derivative_f(x,μ,t)
end

function _param_time_derivative_f(f,x,μ,t,::Any)
  ForwardDiff.gradient(μ->∂t(f)(μ,t)(x),μ)
end

function _param_time_derivative_f(f,x,μ,t,::VectorValue)
  VectorValue(ForwardDiff.gradient(μ->get_array(∂t(f)(μ,t)(x)),μ))
end

function _param_time_derivative_f(f,x,μ,t,::TensorValue)
  TensorValue(ForwardDiff.gradient(μ->get_array(∂t(f)(μ,t)(x)),μ))
end

const ∂μ∂t = param_time_derivative
