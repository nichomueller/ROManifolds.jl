include("functions.jl")
include("networks.jl")
include("Parameters.jl")

using MLJBase
using StableRNGs

"""
    Train the network using the desired architecture that best possible
    matches the training inputs (X) and their corresponding ouptuts(Y)
    over some number of iterations (epochs) and a learning rate (η).
"""
function train_network(layer_dims, X, Y; η=0.001, epochs=1000)
  # Initiate an empty container for cost, iterations, and accuracy at each iteration
  costs = []
  iters = []
  accuracy = []

  # Initialise random weights for the network
  Params = initialise_model_weights(layer_dims)

  # Train the network
  for i = 1:epochs

    Ŷ, caches = forward_propagate_model_weights(X, Params)
    cost = cross_entropy_loss(Ŷ, Y)
    acc = assess_accuracy(Ŷ, Y)
    ∇ = back_propagate_model_weights(Ŷ, Y, caches)
    Params = update_model_weights(Params, ∇, η)

    println("Iteration -> $i, Cost -> $cost, Accuracy -> $acc")

    # Update containers for cost, iterations, and accuracy at the current iteration (epoch)
    push!(iters, i)
    push!(costs, cost)
    push!(accuracy, acc)
  end
  return (cost=costs, iterations=iters, accuracy=accuracy, Parameters=Params)
end

# Generate fake data
X, y = make_blobs(10_000, 3; centers=2, as_table=false, rng=2020);
X = Matrix(X');
y = reshape(y, (1, size(X, 2)));
f(x) = x == 2 ? 0 : x
y2 = f.(y);

# Input dimensions
input_dim = size(X, 1);

# Train the model
nn_results = train_network([input_dim, 5, 3, 1], X, y2; η=0.01, epochs=50);

# Plot accuracy per iteration
p1 = plot(nn_results.accuracy,
  label="Accuracy",
  xlabel="Number of iterations",
  ylabel="Accuracy as %",
  title="Development of accuracy at each iteration");

# Plot cost per iteration
p2 = plot(nn_results.cost,
  label="Cost",
  xlabel="Number of iterations",
  ylabel="Cost (J)",
  color="red",
  title="Development of cost at each iteration");

# Combine accuracy and cost plots
plot(p1, p2, layout=(2, 1), size=(800, 600))
