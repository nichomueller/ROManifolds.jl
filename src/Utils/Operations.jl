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

  sqrt(dot(v, v, X))

end

"""Generate a uniform random vector of dimension n between the ranges set by
  the vector of ranges 'a' and 'b'"""
function generate_parameters(
  a::Vector{T},
  n = 1) where T

  @assert length(a) == 2 "Input vector must be a range, for eg. [0., 1.]"

  [T.(rand(Uniform(a[1], a[2]))) for _ = 1:n]::Vector{T}

end

function generate_parameters(
  a::Vector{Vector{T}},
  n = 1) where T

  Broadcasting(aᵢ -> generate_parameter(aᵢ, n))(a)

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

function solve_cholesky(A::AbstractArray{T}, B::AbstractArray{T}) where T

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] \ B
  x = L[invperm(H.p), :]' \ y

  x

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

  blockmat = Matrix{T}[]
  for nb = 1:nblocks
    push!(blockmat, Mat[:, (nb-1)*ncol_block+1:nb*ncol_block])
  end

  blockmat

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
