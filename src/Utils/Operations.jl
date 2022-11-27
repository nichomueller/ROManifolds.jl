"""Computation of the inner product between 'vec1' and 'vec2', defined by the
  (positive definite) matrix 'norm_matrix'.
  If typeof(norm_matrix) == nothing (default), the standard inner product between
  'vec1' and 'vec2' is returned."""
function LinearAlgebra.dot(
  v1::Vector{T},
  v2::Vector{T},
  X::SparseMatrixCSC) where T

  v1' * X * v2

end

"""Computation of the norm of 'vec', defined by the (positive definite) matrix
'norm_matrix'. If typeof(norm_matrix) == nothing (default), the Euclidean norm
of 'vec' is returned."""
function LinearAlgebra.norm(v::Vector{T}, X::SparseMatrixCSC) where T

  sqrt(LinearAlgebra.dot(v, v, X))

end

function LinearAlgebra.norm(v::Matrix{T}, X::SparseMatrixCSC) where T

  @assert size(v)[2] == 1
  sqrt(LinearAlgebra.dot(v[:,1], v[:,1], X))

end

"""Generate a uniform random vector of dimension n between the ranges set by
  the vector of ranges 'a' and 'b'"""
function generate_parameters(
  a::Vector{Vector{T}},
  n = 1) where T

  function generate_parameter(aᵢ::Vector{T})
    @assert length(aᵢ) == 2 "$aᵢ must be a range, for eg. [0., 1.]"
    T.(rand(Uniform(aᵢ[1], aᵢ[2])))
  end

  [[generate_parameter(a[i]) for i in eachindex(a)] for _ = 1:n]::Vector{Vector{T}}

end

"""Makes use of a truncated SVD (tolerance level specified by 'ϵ') to compute a
  reduced basis 'U' that spans a vector space minimizing the l² distance from
  the vector space spanned by the columns of 'S', the so-called snapshots matrix.
  If the SPD matrix 'X' is provided, the columns of 'U' are orthogonal w.r.t.
  the norm induced by 'X'"""
function POD(S::Matrix{T}, ϵ::Float, X::SparseMatrixCSC{T}) where T

  H = cholesky(X)
  L = sparse(H.L)
  U, Σ, _ = svd(L'*S[H.p, :])

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[N]/energies[end])
  println("Basis number obtained via POD is $N, projection error ≤ $err")

  Matrix{T}((L' \ U[:, 1:N])[invperm(H.p), :])

end

function POD(S::SparseMatrixCSC{T}, ϵ::Float, X::SparseMatrixCSC{T}) where T

  H = cholesky(X)
  L = sparse(H.L)
  U, Σ, _ = svds(L'*S[H.p, :]; nsv=size(S)[2] - 1)[1]

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[N]/energies[end])
  println("Basis number obtained via POD is $N, projection error ≤ $err")

  Matrix{T}((L' \ U[:, 1:N])[invperm(H.p), :])

end

function POD(S::Matrix{T}, ϵ::Float) where T

  U, Σ, _ = svd(S)

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[N]/energies[end])
  println("Basis number obtained via POD is $N, projection error ≤ $err")

  T.(U[:, 1:N])

end

function POD(S::SparseMatrixCSC{T}, ϵ::Float) where T

  U, Σ, _ = svds(S; nsv=size(S)[2] - 1)[1]

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[N]/energies[end])
  println("Basis number obtained via POD is $N, projection error ≤ $err")

  T.(U[:, 1:N])

end

function POD(S::AbstractArray{T}, ϵ::Float, ::Nothing) where T

  POD(S, ϵ)

end

function Gram_Schmidt(
  Vecs::Vector{Vector{T}},
  Φₛ::Matrix{T},
  X₀::SparseMatrixCSC{Float, Int}) where T

  function orthogonalize(vec::Vector{T}, Φ::Matrix{T})
    proj(j::Int) = (dot(vec, Φ[:, j], X₀) / norm(Φ[:, j], X₀)) * Φ[:, j]
    vec - sum(Broadcasting(proj)(1:size(Φ)[2]))
  end

  println("Normalizing primal supremizer 1")
  Vecs[1] = orthogonalize(Vecs[1], Φₛ)
  Vecs[1] /= norm(Vecs[1], X₀)

  for i = 2:length(Vecs)
    println("Normalizing primal supremizer $i")

    Vecs[i] = orthogonalize(Vecs[i], Φₛ)
    Vecs[i] = orthogonalize(Vecs[i], blocks_to_matrix(Vecs[1:i-1]))
    supr_norm = norm(Vecs[i], X₀)

    println("Norm supremizers: $supr_norm")
    Vecs[i] /= supr_norm
  end

  Vecs::Vector{Vector{T}}
end

function solve_cholesky(
  A::SparseMatrixCSC{Float, Int},
  B::SparseMatrixCSC{Float, Int})

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] \ B
  x = L[invperm(H.p), :]' \ y

  x
end

function solve_cholesky(
  A::SparseMatrixCSC{Float, Int},
  B::Matrix{T}) where T

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] \ B
  x = L[invperm(H.p), :]' \ y

  Matrix{T}(x)
end

function solve_cholesky(
  A::SparseMatrixCSC{Float, Int},
  B::Vector{T}) where T

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] \ B
  x = L[invperm(H.p), :]' \ y

  Vector{T}(x)
end

function mode₂_unfolding(Mat₁::Matrix{T}, nₛ::Int) where T
  Nₛ, Nₜnₛ = size(Mat₁)
  Nₜ = Int(Nₜnₛ/nₛ)
  Mat₂ = zeros(T, Nₜ, Nₛ*nₛ)

  function mode₂(i::Int)
    Matrix{T}(Mat₁[:, (i-1)*Nₜ+1:i*Nₜ]')
  end

  blocks_to_matrix(Broadcasting(mode₂)(1:nₛ))::Matrix{T}

end

function mode₂_unfolding(Mat₁::Vector{Matrix{T}}, nₛ::Int) where T
  Broadcasting(mat₁ -> mode₂_unfolding(mat₁, nₛ))(Mat₁)::Vector{Matrix{T}}
end

function vector_to_matrix(
  Vec::Vector{T},
  r::Int,
  c::Int) where T

  @assert length(Vec) == r*c "Wrong dimensions"
  Matrix{T}(reshape(Vec, r, c))
end

function vector_to_matrix(
  Vec::Vector{Vector{T}},
  r::Vector{Int},
  c::Vector{Int}) where T

  Broadcasting(vector_to_matrix)(Vec, r, c)
end

function blocks_to_matrix(Mat_block::Vector{Matrix{T}}) where T

  for i = 1 .+ eachindex(Mat_block[2:end])
    @assert size(Mat_block[i])[1] == size(Mat_block[1])[1] "Rows in Mat_block must be the same"
  end

  Matrix{T}(reduce(vcat, transpose.(Mat_block))')

end

function blocks_to_matrix(Vec_block::Vector{Vector{T}}) where T

  Matrix{T}(reduce(vcat, transpose.(Vec_block))')

end

function blocks_to_matrix(Mat_block::Vector{SparseMatrixCSC{T,Int}}) where T

  for i = 1 .+ eachindex(Mat_block[2:end])
    @assert size(Mat_block[i])[1] == size(Mat_block[1])[1] "Rows in Mat_block must be the same"
  end

  sparse(reduce(vcat, transpose.(Mat_block))')

end

function blocks_to_matrix(Vec_block::Vector{Vector{Vector{T}}}) where T
  n = length(Vec_block)
  mat = blocks_to_matrix(Vec_block[1])
  if n > 1
    for i = 2:n
      mat = hcat(mat,blocks_to_matrix(Vec_block[i]))
    end
  end
  mat
end

function matrix_to_vecblocks(Mat::Matrix{T}) where T
  [Mat[:, i] for i = 1:size(Mat)[2]]::Vector{Vector{T}}
end

function matrix_to_blocks(Mat::Matrix{T}, nblocks=nothing) where T

  if isnothing(nblocks)
    nblocks = size(Mat)[2]
  else
    @assert size(Mat)[2] % nblocks == 0 "Something is wrong"
  end

  ncol_block = Int(size(Mat)[2] / nblocks)
  idx2 = ncol_block:ncol_block:size(Mat)[2]
  idx1 = idx2 .- ncol_block .+ 1

  blockmat = Matrix{T}[]
  for i in eachindex(idx1)
    push!(blockmat, Mat[:, idx1[i]:idx2[i]])
  end

  blockmat::Vector{Matrix{T}}

end

function matrix_to_blocks(Mat::Array{T, 3}) where T

  blockmat = Matrix{T}[]
  for nb = 1:size(Mat)[end]
    push!(blockmat, Mat[:, :, nb])
  end

  blockmat

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

function Base.NTuple(N::Int, T::DataType)

  NT = ()
  for _ = 1:N
    NT = (NT...,zero(T))
  end

  NT::Base.NTuple{N,T}

end

Gridap.VectorValue(D::Int, T::DataType) = VectorValue(NTuple(D, T))

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
