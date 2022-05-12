"""
    Sigmoid activation function
"""
function sigmoid(z)
    y = 1 ./ (1 .+ exp.(.-z))
    return (y = y, z = z)
end

"""
    Derivative of the Sigmoid function.
"""
function sigmoid_backwards(∂y, activated_cache)
    y = sigmoid(activated_cache).y
    ∂z = ∂y .* y .* (1 .- y)

    @assert (size(∂z) == size(activated_cache))
    return ∂z
end

"""
    ReLU activation function
"""
function relu(z)
    y = max.(0, z)
    return (y = y, z = z)
end

"""
    Derivative of the ReLU function.
"""
function relu_backwards(∂y, activated_cache)
    return ∂y .* (activated_cache .> 0)
end

"""
    Computes the log loss (binary cross entropy) of the current predictions.
"""
function cross_entropy_loss(Ŷ, Y)
    m = size(Y, 2)
    epsilon = eps(1.0)

    # Deal with log(0) scenarios
    Ŷ_new = [max(i, epsilon) for i in Ŷ]
    Ŷ_new = [min(i, 1-epsilon) for i in Ŷ_new]

    return -sum(Y .* log.(Ŷ_new) + (1 .- Y) .* log.(1 .- Ŷ_new)) / m
end

"""
    Derivative of the cross entropy loss.
"""
function ∂cross_entropy_loss(Ŷ, Y)

  epsilon = eps(1.0)

  # Deal with 1/0 scenarios
  Ŷ_new = [max(i, epsilon) for i in Ŷ]
  Ŷ_new = [min(i, 1-epsilon) for i in Ŷ_new]

  return -(Y ./ Ŷ_new) .+ ((1 .- Y) ./ ( 1 .- Ŷ_new))
end

"""
    Computes the MSE loss of the current predictions.
"""
function MSE_loss(Ŷ, Y)
  m = size(Y, 2)

  return sum((Ŷ-Y).^2) / m
end

"""
    Derivative of the MSE loss.
"""
function ∂MSE_loss(Ŷ, Y)

  return 2*(Ŷ-Y)

end

"""
    Check the accuracy between predicted values (Ŷ) and the true values(Y).
"""
function assess_accuracy(Ŷ , Y)
    @assert size(Ŷ) == size(Y)
    return sum((Ŷ .> 0.5) .== Y) / length(Y)
end
