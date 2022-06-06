"""
    Funtion to initialise the Parameters or weights of the desired network.
"""
function initialise_model_weights(layer_dims)
    Params = Dict()

    # Build a dictionary of initialised weights and bias units
    for l=2:length(layer_dims)
        Params[string("W_", (l-1))] = rand(StableRNG(1), layer_dims[l], layer_dims[l-1]) * sqrt(2 / (layer_dims[l-1]+layer_dims[l]))
        Params[string("b_", (l-1))] = zeros(layer_dims[l], 1)
    end

    return Params
end

"""
    Partial derivatives of the components of linear forward function
    using the linear output (∂Z) and caches of these components (cache).
"""
function linear_backward(∂zₙ, cache)
    # Unpack cache
    xₙ , W , b = cache
    m = size(xₙ, 2)

    # Partial derivates of each of the components
    ∂W = ∂zₙ * (xₙ') / m
    ∂b = sum(∂zₙ, dims = 2) / m
    ∂xₙ = (W') * ∂zₙ

    @assert (size(∂xₙ) == size(xₙ))
    @assert (size(∂W) == size(W))
    @assert (size(∂b) == size(b))

    return ∂W , ∂b , ∂xₙ
end

"""
    Unpack the linear activated caches (cache) and compute their derivatives
    from the applied activation function.
"""
function linear_activation_backward(∂yₙ, cache, activation_function="relu")
    @assert activation_function ∈ ("sigmoid", "relu")

    linear_cache , cache_activation = cache

    if (activation_function == "relu")

        ∂zₙ = relu_backwards(∂yₙ, cache_activation)
        ∂W , ∂b , ∂xₙ = linear_backward(∂zₙ, linear_cache)

    elseif (activation_function == "sigmoid")

        ∂zₙ = sigmoid_backwards(∂yₙ, cache_activation)
        ∂W , ∂b , ∂xₙ = linear_backward(∂zₙ, linear_cache)

    end

    return ∂W , ∂b , ∂xₙ
end

"""
    Compute the gradients (∇) of the Parameters (master_cache) of the constructed model
    with respect to the cost of predictions (Ŷ) in comparison with actual output (Y).
"""
function back_propagate_model_weights(Ŷ, Y, master_cache)
    # Initiate the dictionary to store the gradients for all the components in each layer
    ∇ = Dict()

    L = length(master_cache)
    Y = reshape(Y , size(Ŷ))

    # Partial derivative of the output layer
    ∂Ŷ = -(Y ./ Ŷ) .+ ((1 .- Y) ./ ( 1 .- Ŷ))
    current_cache = master_cache[L]

    # Backpropagate on the layer preceeding the output layer
    ∇[string("∂W_", (L))], ∇[string("∂b_", (L))], ∇[string("∂x_", (L-1))] = linear_activation_backward(∂Ŷ, current_cache, "sigmoid")
    # Go backwards in the layers and compute the partial derivates of each component.
    for l=reverse(0:L-2)
        current_cache = master_cache[l+1]
        ∇[string("∂W_", (l+1))], ∇[string("∂b_", (l+1))], ∇[string("∂x_", (l))] = linear_activation_backward(∇[string("∂x_", (l+1))], current_cache, "relu")
    end

    # Return the gradients of the network
    return ∇
end

"""
    Update the Paramaters of the model using the gradients (∇)
    and the learning rate (η).
"""
function update_model_weights(Parameters, ∇, η)

    L = Int(length(Parameters) / 2)

    # Update the Parameters (weights and biases) for all the layers
    for l = 0: (L-1)
        Parameters[string("W_", (l + 1))] -= η .* ∇[string("∂W_", (l + 1))]
        Parameters[string("b_", (l + 1))] -= η .* ∇[string("∂b_", (l + 1))]
    end

    return Parameters
end
