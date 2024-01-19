function ODETools.∂t(f::TransientPFunction)
  @unpack fun,params,times = f
  function ∂ₚt(x,μ,t)
    fxt = zero(return_type(fun,x,μ,t))
    _∂ₚt(fun,x,μ,t,fxt)
  end
  ∂ₚt(x::VectorValue) = (μ,t) -> ∂ₚt(x,μ,t)
  ∂ₚt(μ,t) = x -> ∂ₚt(x,μ,t)
  return TransientPFunction(∂ₚt,params,times)
end

function _∂ₚt(f,x,μ,t,::Any)
  derivative(t->f(μ,t)(x),t)
end

function _∂ₚt(f,x,μ,t,::VectorValue)
  VectorValue(derivative(t->get_array(f(μ,t)(x)),t))
end

function _∂ₚt(f,x,μ,t,::TensorValue)
  TensorValue(derivative(t->get_array(f(μ,t)(x)),t))
end
