function train_network(layer_dims, X, Y; η=0.001, epochs=1000)
  # Initiate an empty container for cost, iterations, and accuracy at each iteration
  costs = []
  iters = []
  accuracy = []

  # Initialise random weights for the network
  Params = initialise_model_weights(layer_dims)

  # Train the network
  for i = 1:n_epochs

    for nb = 1:n_batches

      X_nb = X[nb]
      Ŷ = model(X_nb, Params)
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

  end

  return (cost=costs, iterations=iters, accuracy=accuracy, Parameters=Params)

end

###############################################################################
train_x, train_y = CIFAR10.traindata(Float32)
labels = onehotbatch(train_y, 0:9)

train = ([(train_x[:,:,:,i], labels[:,i]) for i in partition(1:49000, 1000)]) |> gpu
valset = 49001:50000
valX = train_x[:,:,:,valset] |> gpu
valY = labels[:, valset] |> gpu

Net = Chain(
  Conv((5,5), 3=>16, relu),
  MaxPool((2,2)),
  Conv((5,5), 16=>8, relu),
  MaxPool((2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(200, 120),
  Dense(120, 84),
  Dense(84, 10),
  softmax) |> gpu

#loss(x, y) = sum(crossentropy(Net(x), y))
opt = Momentum(0.01)

accuracy(x, y) = mean(onecold(Net(x), 0:9) .== onecold(y, 0:9))

epochs = 10

for epoch = 1:epochs
  for d in train
    gs = gradient(Params(Net)) do
      l = loss(d...)
    end
    update!(opt, Params(Net), gs)
  end
  @show accuracy(valX, valY)
end

test_x, test_y = CIFAR10.testdata(Float32)
test_labels = onehotbatch(test_y, 0:9)
test = gpu.([(test_x[:,:,:,i], test_labels[:,i]) for i in partition(1:10000, 1000)])

#= ids = rand(1:10000, 5)
rand_test = test_x[:,:,:,ids] |> gpu
rand_truth = test_y[ids]
Net(rand_test) =#

class_correct = zeros(10)
class_total = zeros(10)
for i in 1:10
  preds = m(test[i][1])
  lab = test[i][2]
  for j = 1:1000
    pred_class = findmax(preds[:, j])[2]
    actual_class = findmax(lab[:, j])[2]
    if pred_class == actual_class
      class_correct[pred_class] += 1
    end
    class_total[actual_class] += 1
  end
end

class_correct ./ class_total

################################################################################
using Dates
first_train_epoch =  datetime2unix(DateTime("2010-01-01T00:00:00"))
last_train_epoch = datetime2unix(DateTime("2020-12-31T23:59:59"))
last_test_epoch = datetime2unix(DateTime("2030-12-31T23:59:59"))

y = rand(first_train_epoch:1.0:last_train_epoch, 1000)

function y_normaliser(y)
  (y - first_train_epoch) / (last_test_epoch - first_train_epoch)
end

# transform our predictions back to the correct scale
# y_rescaler(y_normaliser(y)) == y
function y_rescaler(y_predict)
  y_predict * (last_test_epoch - first_train_epoch) + first_train_epoch
end


# split x's into year, month, day, hour, minute, seconds as inputs
function datetime2input(x)
firstyear = year(unix2datetime(first_train_epoch))
lastyear = year(unix2datetime(last_test_epoch))
  Float64.([
    (year(x) - firstyear) / (lastyear - firstyear),
    month(x) / 12,
    day(x) / 31,
    hour(x) / 24,
    minute(x) / 60,
    second(x) / 60
])
end

batch_size = 10
xs = hcat.(partition(datetime2input.(unix2datetime.(y)), batch_size)...)
ys = hcat.(partition(y_normaliser.(y), batch_size)...)

# single layer NN
# we have 6 inputs: year, month, day, hour, minute, second
model = Dense(6, 1, identity)

# try different loss functions, I use mean squared error here
Loss(x, y) = Flux.mse(model(x), y)

# you can use different optimisers too!
optimizer = ADAM()
optimizer1 = LBFGS()

training_loss = Float64[]
epochs = Int64[]

for epoch in 1:1000
	train!(Loss, Params(model), zip(xs, ys), optimizer)

	if epoch % 10 == 0
	  # we record our training loss
		push!(epochs, epoch)
		push!(training_loss, sum(Loss.(xs, ys)))
	end
end


# plot the results
plot(epochs, training_loss; title="Training loss 2010-2020")
