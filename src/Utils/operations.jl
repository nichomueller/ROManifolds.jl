const F = Function
const Float = Float64

"""Computation of the inner product between 'vec1' and 'vec2', defined by the
  (positive definite) matrix 'norm_matrix'.
  If typeof(norm_matrix) == nothing (default), the standard inner product between
  'vec1' and 'vec2' is returned."""
function mydot(vec1::Vector{T}, vec2::Vector{T}, norm_matrix::SparseMatrixCSC) where T

  sum(sum(vec1' * norm_matrix * vec2))

end

function mydot(vec1::Vector{T}, vec2::Vector{T}) where T

  sum(sum(vec1' * vec2))

end

"""Computation of the norm of 'vec', defined by the (positive definite) matrix
'norm_matrix'. If typeof(norm_matrix) == nothing (default), the Euclidean norm
of 'vec' is returned."""
function mynorm(vec::Vector{T}, norm_matrix::SparseMatrixCSC) where T

  sqrt(mydot(vec, vec, norm_matrix))

end

function mynorm(vec::Vector{T}) where T

  sqrt(mydot(vec, vec))

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
function generate_parameter(a::Vector{T}, b::Vector{T}, n = 1) where T
  [[T.(rand(Uniform(a[i], b[i]))) for i = eachindex(a)] for j in 1:n]::Vector{Vector{T}}
end

"""Makes use of a truncated SVD (tolerance level specified by 'ϵ') to compute a
  reduced basis 'U' that spans a vector space minimizing the l² distance from
  the vector space spanned by the columns of 'S', the so-called snapshots matrix.
  If the SPD matrix 'X' is provided, the columns of 'U' are orthogonal w.r.t.
  the norm induced by 'X'"""
function POD(S::Matrix{T}, ϵ::Float, X::SparseMatrixCSC{T}) where T

  H = cholesky(X)
  L = sparse(H.L)
  #mul!(S, L', S[H.p, :])
  U, Σ, _ = svd(L'*S[H.p, :])

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  println("Basis number obtained via POD is $N, projection error ≤ $(sqrt(1-energies[N]/energies[end]))")

  Matrix{T}((L' \ U[:, 1:N])[invperm(H.p), :])

end

function POD(S::SparseMatrixCSC{T}, ϵ::Float, X::SparseMatrixCSC{T}) where T

  H = cholesky(X)
  L = sparse(H.L)
  #mul!(S, L', S[H.p, :])
  U, Σ, _ = svds(L'*S[H.p, :]; nsv=size(S)[2] - 1)[1]

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  println("Basis number obtained via POD is $N, projection error ≤ $(sqrt(1-energies[N]/energies[end]))")

  Matrix{T}((L' \ U[:, 1:N])[invperm(H.p), :])

end

function POD(S::Matrix{T}, ϵ::Float) where T

  U, Σ, _ = svd(S)

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  println("Basis number obtained via POD is $N, projection error ≤ $(sqrt(1-energies[N]/energies[end]))")

  T.(U[:, 1:N])

end

function POD(S::SparseMatrixCSC, ϵ::Float) where T

  U, Σ, _ = svds(S; nsv=size(S)[2] - 1)[1]

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  println("Basis number obtained via POD is $N, projection error ≤ $(sqrt(1-energies[N]/energies[end]))")

  T.(U[:, 1:N])

end

function mode₂_unfolding(Mat₁::Matrix{T}, nₛ::Int) where T
  Nₛ, Nₜnₛ = size(Mat₁)
  Nₜ = Int(Nₜnₛ/nₛ)
  Mat₂ = zeros(T, Nₜ, Nₛ*nₛ)
  @simd for i in 1:nₛ
    Mat₂[:, (i-1)*Nₛ+1:i*Nₛ] = Mat₁[:, (i-1)*Nₜ+1:i*Nₜ]'
  end
  Mat₂
end

function newton(
  res::F,
  Jₙ::Matrix{T},
  u₀::Vector{T},
  ϵ::Float,
  max_k::Int) where T

  k = 0
  uᵏ = u₀
  while k ≤ max_k && norm(res(uᵏ)) ≥ ϵ
    uᵏ -= Jₙ \ res(uᵏ)::Vector{T}
    k += 1
  end

  uᵏ::Vector{T}

end

function get_NTuple(N::Int, T::DataType)

  ntupl = ()
  for _ = 1:N
    ntupl = (ntupl...,zero(T))
  end

  ntupl::NTuple{N,T}

end

Gridap.VectorValue(D::Int, T::DataType) = VectorValue(get_NTuple(D, T))

function Base.one(vv::VectorValue{D,T}) where {D,T}
  vv_one = zero(vv) .+ one(T)
  vv_one::VectorValue{D,T}
end

function Base.Float64(vv::VectorValue{D,Float}) where D
  VectorValue(Float64.([vv...]))
end

function Base.Float32(vv::VectorValue{D,Float32}) where D
  VectorValue(Float32.([vv...]))
end

function Base.Int64(vv::VectorValue{D,Int}) where D
  VectorValue(Int64.([vv...]))
end

function Base.Int32(vv::VectorValue{D,Int32}) where D
  VectorValue(Int32.([vv...]))
end
