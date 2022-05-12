"""
    Make a linear forward calculation
"""
function linear_forward(x, W, b)
  # Make a linear forward and return inputs as cache
  z = (W * x) .+ b
  cache = (x, W, b)

  @assert size(z) == (size(W, 1), size(x, 2))

  return (z=z, cache=cache)
end

"""
    Make a forward activation from a linear forward.
"""
function linear_forward_activation(xₙ, W, b, activation_function="relu")
  @assert activation_function ∈ ("sigmoid", "relu")
  zₙ, linear_cache = linear_forward(xₙ, W, b)

  if activation_function == "sigmoid"
    yₙ, activation_cache = sigmoid(zₙ)
  end

  if activation_function == "relu"
    yₙ, activation_cache = relu(zₙ)
  end

  cache = (linear_step_cache=linear_cache, activation_step_cache=activation_cache)

  @assert size(yₙ) == (size(W, 1), size(xₙ, 2))

  return yₙ, cache
end

"""
    Forward the design matrix through the network layers using the parameters.
"""
function forward_propagate_model_weights(X, parameters)
  master_cache = []
  yₙ = X
  L = Int(length(parameters) / 2)

  # Forward propagate until the last (output) layer
  for l = 1:(L-1)
    xₙ = yₙ
    yₙ, cache = linear_forward_activation(xₙ, parameters[string("W_", (l))], parameters[string("b_", (l))], "relu")
    push!(master_cache, cache)
  end

  # Make predictions in the output layer
  Ŷ, cache = linear_forward_activation(yₙ, parameters[string("W_", (L))], parameters[string("b_", (L))], "sigmoid")
  push!(master_cache, cache)

  return Ŷ, master_cache
end
