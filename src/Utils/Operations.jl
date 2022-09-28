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

function POD(S::SparseMatrixCSC{T}, ϵ::Float) where T

  U, Σ, _ = svds(S; nsv=size(S)[2] - 1)[1]

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  println("Basis number obtained via POD is $N, projection error ≤ $(sqrt(1-energies[N]/energies[end]))")

  T.(U[:, 1:N])

end

function solve_cholesky(A::SparseMatrixCSC{T}, b::Vector{T}) where T

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] \ b
  x = L[invperm(H.p), :]' \ y

  x::Vector{T}

end

function solve_cholesky(A::SparseMatrixCSC{T}, B::Matrix{T}) where T

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] * B
  x = L[invperm(H.p), :]' \ y

  x::Matrix{T}

end

function solve_cholesky(A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T}) where T

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] * B
  x = L[invperm(H.p), :]' \ y

  x::Matrix{T}

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

function blocks_to_matrix(Mat_block::Vector{Matrix{T}}) where T

  Matrix{T}(reduce(vcat, transpose.(Mat_block))')

end

function blocks_to_matrix(Vec_block::Vector{Vector{T}}) where T

  Matrix{T}(reduce(vcat, transpose.(Vec_block))')

end

function matrix_to_blocks(Mat::Matrix{T}, nblocks::Int) where T

  @assert size(Mat)[2] % nblocks == 0 "Something is wrong"
  ncol_block = Int(size(Mat)[2] / nblocks)

  Mat_new = Matrix{T}[]
  for nb = 1:nblocks
    push!(Mat_new, Mat[:, (nb-1)*ncol_block+1:nb*ncol_block])
  end

  Mat_new

end

function SparseArrays.findnz(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

  numnz = nnz(S)
    I = Vector{Ti}(undef, numnz)
    J = Vector{Ti}(undef, numnz)
    V = Vector{Tv}(undef, numnz)

    count = 1
    @inbounds for col = 1 : size(S, 2), k = SparseArrays.getcolptr(S)[col] : (SparseArrays.getcolptr(S)[col+1]-1)
        I[count] = rowvals(S)[k]
        J[count] = col
        V[count] = nonzeros(S)[k]
        count += 1
    end

  nz = findall(x -> x .!= 0., V)

  (I[nz], J[nz], V[nz])

end

function SparseArrays.findnz(x::SparseVector{Tv,Ti}) where {Tv,Ti}

  numnz = nnz(x)

    I = Vector{Ti}(undef, numnz)
    V = Vector{Tv}(undef, numnz)

    nzind = SparseArrays.nonzeroinds(x)
    nzval = nonzeros(x)

    @inbounds for i = 1 : numnz
        I[i] = nzind[i]
        V[i] = nzval[i]
    end

  nz = findall(v -> v .!= 0., V)

  (I[nz], V[nz])

end

function newton(
  res::Function,
  J::Matrix{T},
  u₀,
  ϵ::Float,
  max_k::Int) where T

  k = 0
  uᵏ = u₀
  while k ≤ max_k && norm(res(uᵏ)) ≥ ϵ
    uᵏ -= J \ res(uᵏ)::Vector{T}
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
