struct Opt
  rank::Vector{Int64}
  epsilon::Vector{Float64}
end

struct TensorTrain
  opt::Opt
  norms::Vector
  cores::Vector
  ranks::Vector{Int64}
  singular::Vector{Float64}
end

function TensorTrain(X; opt::Opt = Opt(30, 0.0), norms = fill(I, ndims(X)))
(cores, ranks, singular) = ttsvd(X, norms, opt)
return TensorTrain(opt, norms, cores, ranks, singular)
end

function Base.length(tt::TensorTrain)
  Base.length(tt.cores)
end

function Base.display(tt::TensorTrain)
  println("A tensor train of length of $(length(tt))")
  println("the ranks are $(tt.ranks)")
end

function Base.size(tt::TensorTrain)
  s = [Base.size(tt.cores[k], 2) for k = 1:length(tt)]
end

function proj(A, core::Array{Float64, 3}, X)
  if size(core)[1:2] != size(A)[1:2] error("The dimension of subspace does not match the array") end
  U = reshape(core, :, size(core, 3))
  if X == I
      Xu = I
  else
      Xu = kronecker(X, I(size(core, 1)))
  end
  n_3 = size(A)[3:end]
  n_new = (size(U, 2), n_3...)
  return reshape(U'*Xu*reshape(A, size(A, 1)*size(A, 2), :), n_new)
end

function proj(A, tt::TensorTrain, k::Integer)
  # check the diemension
  if ndims(A) != length(tt) error("The input array and tensor train should have same length") end
  if !all(size(A) .== size(tt)) error("The input array and tensor train should have same size") end
  if !(1 <= k <= length(tt)) error("The input index k is not in the range") end

  A = reshape(A, (1, size(A)...))
  for n = 1:k
      A = proj(A, tt.cores[n], tt.norms[n])
  end
  return A
end

function eval(tt::TensorTrain, ind::Vector{<:Integer})
  # check if ind is of correct length
  if length(ind) != length(tt)
      error("The index length does not match the tensor train")
  end
  if !all(0 .< ind .<= size(tt))
      error("The requested index exceeds the tensor train size")
  end
  output = 1
  for k = 1:length(tt)
      C = tt.cores[k]
      output = output * C[:, ind[k], :]
  end
  return output[1]
end

function basis(tt::TensorTrain, k::Integer)
  # extract the basis vectors for the k-th core
  if ! (0 < k <= length(tt)) error("Index exceeds the tensor train length") end
  A = tt.cores[k]
  A = [A[i1, :, i2] for i1 in 1:size(tt.cores[k], 1), i2 in 1:size(tt.cores[k], 3)]
end


function basis(tt::TensorTrain, k1::Integer, k2::Integer)
  # extract the product basis vectors from the k1-the core to the k2-th core
  if ! (0 < k1 < k2 <= length(tt)) error("Indexes exceed the tensor train length or are not in ascending order") end
  A = basis(tt, k1)
  for k = k1:k2-1
      A = A*basis(tt, k+1)
  end
  return A
end

function base2mat(A::Matrix)
  # arrange the basis in a matrix of size n × r.
  reduce(hcat, reshape(A, 1,:))
end

function Base.:*(x::Vector{Float64}, y::Vector{Float64})
  z = kronecker(y, x)
  return z[:]
end

function ttsvd(X, norms, opt)
  # check opt matches the dimension of X
  if length(opt.rank) == 1
      optrank = fill(opt.rank[1], ndims(X) - 1)
  elseif length(opt.rank) < ndims(X) - 1
      error("The input ranks do not match tensor dimension")
  else
      optrank = opt.rank
  end
  if length(opt.epsilon) == 1
      optepsilon = fill(opt.epsilon[1], ndims(X) - 1)
  elseif length(opt.epsilon) < ndims(X) - 1
      error("The input epsilon do not match tensor dimension")
  else
      optepsilon = opt.epsilon
  end

  d = ndims(X)
  n = size(X)
  delta = opt.epsilon/sqrt(ndims(X)-1) * norm(X)

  # initialization
  singular = zeros(d)
  ranks = fill(1,d)
  cores = Vector(undef, ndims(X))
  T = X

  for k = 1:d-1
      M = norms[k]
      if M != I; M = cholesky(M, check = true).U end
      if k == 1
          X_k = reshape(nprod(T, M, 1), n[k], :)
      else
          X_k = reshape(nprod(T, M, 2), ranks[k-1] * n[k], :)
      end

      U,S,V = svd(X_k)
      n2 = sqrt(sum(S.^2))
      r = min(optrank[k], findlast(S .> n2*optepsilon[k]))
      U = U[:, 1:r]
      S = S[1:r]
      V = V[:, 1:r]

      singular[k] =  r < length(S) ? sum(S[r+1:end])/n2 : 0
      ranks[k] = r
      T = reshape(S.*V', ranks[k], n[k+1], :)
      if k == 1
          cores[k] = nprod(reshape(U, 1, n[k], ranks[k]), inv(M), 2)
      else
          cores[k] = nprod(reshape(U, ranks[k-1], n[k], ranks[k]), inv(M), 2)
      end
  end
  cores[d] = reshape(T, ranks[d-1], n[d], 1)
  return cores, ranks, singular
end

function nprod(A, M, k)
  if M == I
      return A
  end
  d = ndims(A)
  perm = circshift(1:d, -(k-1))
  A = permutedims(A, perm)
  n = size(A)
  n = (size(M, 1), n[2:d]...)
  A = reshape(M*reshape(A, size(A, 1), :), n)
  permutedims(A, invperm(perm))
end
