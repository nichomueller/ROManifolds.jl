function define_network(net_number::Int)
  if net_number === 1
    return Chain(
      Conv((5,5), 3=>16, relu),
      MaxPool((2,2)),
      Conv((5,5), 16=>8, relu),
      MaxPool((2,2)),
      x -> reshape(x, :, size(x, 4)),
      Dense(200, 120),
      Dense(120, 84),
      Dense(84, 10),
      softmax) |> gpu
  elseif net_number === 2
    return Chain(
      Conv((5,5), 1=>8, elu; stride = 2, pad = SamePad()),
      Conv((5,5), 8=>16, elu; stride = 2, pad = SamePad()),
      Conv((5,5), 16=>32, elu; stride = 2, pad = SamePad()),
      Conv((5,5), 32=>64, elu; stride = 2, pad = SamePad()),
      Dense(512, 512),
      Dense(512, 512),
      Conv((5,5), 64=>32, elu; stride = 2, pad = SamePad()),
      Conv((5,5), 32=>16, elu; stride = 2, pad = SamePad()),
      Conv((5,5), 16=>8, elu; stride = 2, pad = SamePad()),
      Conv((5,5), 8=>1, identity; stride = 2, pad = SamePad())
      ) |> gpu
  end
end
