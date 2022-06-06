η = 0.01
n_epochs = 50
n_batches = 10
validation_percent = 0.1
activation = "ReLU"
architecture = "1"
loss = "C-E"
optimizer = "LBFGS"
shuffle_input = true
batch_normalization = false
early_stopping = true

mutable struct DLSpecifics{T<:String}
  activation::T
  architecture::T
  loss::T
  optimizer::T
end

DL_info = DLSpecifics(activation, architecture, loss, optimizer)
