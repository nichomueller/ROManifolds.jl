
"""Computation of the inner product between 'vec1' and 'vec2', defined by the
  (positive definite) matrix 'norm_matrix'.
  If typeof(norm_matrix) == nothing (default), the standard inner product between
  'vec1' and 'vec2' is returned."""
function mydot(vec1::Vector, vec2::Vector, norm_matrix = nothing)

  if isnothing(norm_matrix)
      norm_matrix = float(I(size(vec1)[1]))
  end
  sum(sum(vec1' * norm_matrix * vec2))

end

"""Computation of the norm of 'vec', defined by the (positive definite) matrix
'norm_matrix'. If typeof(norm_matrix) == nothing (default), the Euclidean norm
of 'vec' is returned."""
function mynorm(vec::Vector, norm_matrix = nothing)

  if isnothing(norm_matrix)
    norm_matrix = float(I(size(vec)[1]))
  end
  sqrt(mydot(vec, vec, norm_matrix))

end

"""Computation of the standard matrix product between A, tensor of order 3,
  and B, tensor of order 2. If transpose_A is true, then the first two dimensions
  of A are permuted"""
function matrix_product!(
    AB::Array, A::Array, B::Array; transpose_A=false)
  @assert length(size(A)) == 3 && length(size(B)) == 2 "Only implemented tensor order 3 * tensor order 2"
  if transpose_A
    A = permutedims(A,[2,1,3])
  end
  for q = 1:size(A)[3]
    AB[:,:,q] = A[:,:,q]*B
  end
  AB
end

"""Computation of the standard matrix product between A, tensor of order 3,
  and B, tensor of order 2. If transpose_A is true, then the first two dimensions
  of A are permuted"""
function matrix_product_vec!(
    AB::Array, A::Array, B::Array; transpose_A=false)
  @assert length(size(A)) == 3 && length(size(B)) == 2 || length(size(B)) == 3
   "Only implemented tensor order 3 * tensor order 2 or 3"
  if transpose_A
    A = permutedims(A,[2,1,3])
  end
  if length(size(B)) == 2
    for q₁ = 1:size(A)[end]
      for q₂ = 1:size(B)[end]
        AB[:,:,q₂+(q₁-1)*size(B)[end]] = A[:,:,q₁]*B[:,q₂]
      end
    end
  else
    for q₁ = 1:size(A)[end]
      for q₂ = 1:size(B)[end]
        AB[:,:,q₂+(q₁-1)*size(B)[end]] = A[:,:,q₁]*B[:,:,q₂]
      end
    end
  end
  AB
end

"""Generate a uniform random vector of dimension n between the ranges set by
  the vector of ranges 'a' and 'b'"""
function generate_parameter(a::Vector, b::Vector, n::Int64 = 1)

  return [[rand(Uniform(a[i], b[i])) for i = eachindex(a)] for j in 1:n]

end

const GeneralizedMatrix = Union{Matrix,SparseMatrixCSC}
const GeneralizedVector = Union{Vector,SparseVector}
const GeneralizedMatVec = Union{GeneralizedMatrix,GeneralizedVector}

"""Makes use of a truncated SVD (tolerance level specified by 'ϵ') to compute a
  reduced basis 'U' that spans a vector space minimizing the l² distance from
  the vector space spanned by the columns of 'S', the so-called snapshots matrix.
  If the SPD matrix 'X' is provided, the columns of 'U' are orthogonal w.r.t.
  the norm induced by 'X'"""
function POD(S::GeneralizedMatrix, ϵ::Float64=1e-5, X=nothing)
  S̃ = copy(S)
  if !isnothing(X)
    if !issparse(X)
      X = sparse(X)
    end
    H = cholesky(X)
    L = sparse(H.L)
    mul!(S̃, L', S̃[H.p, :])
  end
  if issparse(S̃)
    U, Σ, _ = svds(S̃; nsv=size(S̃)[2] - 1)[1]
  else
    U, Σ, _ = svd(S̃)
  end

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  @info "Basis number obtained via POD is $N,
    projection error ≤ $(sqrt(1-energies[N]/energies[end]))"
  if issparse(U)
    U = Matrix(U)
  end
  if !isnothing(X)
    return Matrix((L' \ U[:, 1:N])[invperm(H.p), :]), Σ
  else
    return U[:,1:N], Σ
  end

end
