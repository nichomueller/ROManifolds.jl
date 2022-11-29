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

get_Nt(S::AbstractMatrix,ns::Int) = Int(size(S,2)/ns)
function mode2_unfolding(S::AbstractMatrix,ns::Int)
  Nt = get_Nt(S,ns)
  idx_fun(ns) = (ns .- 1)*Nt .+ 1:ns*Nt
  idx = idx_fun.(1:ns)
  mode2 = Matrix.(transpose.(S[idx]))

  blocks_to_matrix(mode2)
end

my_svd(s::Matrix) = svd(s)
my_svd(s::SparseMatrixCSC) = svds(s;nsv=size(S)[2]-1)[1]
my_svd(s::Vector{AbstractMatrix}) = my_svd(blocks_to_matrix(s))

function POD(S::AbstractMatrix,ϵ=1e-5)
  U, Σ, _ = my_svd(S)
  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[N]/energies[end])
  println("Basis number obtained via POD is $N, projection error ≤ $err")

  U[:, 1:N]
end

function POD(S::AbstractMatrix,X::SparseMatrixCSC,ϵ=1e-5)
  H = cholesky(X)
  L = sparse(H.L)
  U, Σ, _ = my_svd(L'*S[H.p, :])

  energies = cumsum(Σ.^2)
  N = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[N]/energies[end])
  println("Basis number obtained via POD is $N, projection error ≤ $err")

  Matrix((L' \ U[:, 1:N])[invperm(H.p), :])
end

projection(vnew::AbstractVector,v::AbstractVector) = v*(vnew'*v)
projection(vnew::AbstractVector,basis::AbstractMatrix) =
  sum([projection(vnew,basis[:,i]) for i=axis(basis,2)])
orth_projection(vnew::AbstractVector,v::AbstractVector) = projection(vnew,v)/(v'*v)
orth_projection(vnew::AbstractVector,basis::AbstractMatrix) =
  sum([orth_projection(vnew,basis[:,i]) for i=axis(basis,2)])

isbasis(basis,args...) =
  all([isapprox(norm(basis[:,j],args...),1) for j=axes(basis,2)])

function orth_complement(
  v::AbstractVector{T},
  basis::AbstractMatrix{T}) where T

  @assert isbasis(basis) "Provide a basis"
  v - projection(v,basis)
end

function gram_schmidt!(vec::Matrix,basis::Vector)

  println("Normalizing primal supremizer 1")
  vec[:,1] = orth_complement(vec[:,1],basis)

  for i = 2:size(vec,2)
    println("Normalizing primal supremizer $i")
    vec[:,i] = orth_complement(vec[:,i],basis)
    vec[:,i] = orth_complement(vec[:,i],vec[:,1:i-1])
    supr_norm = norm(vec[:,i])
    println("Norm supremizers: $supr_norm")
    vec[:,i] /= supr_norm
  end

  vec::Vector{Vector{T}}
end

function gram_schmidt!(vec::Vector{Vector},basis::Matrix)
  gram_schmidt!(blocks_to_matrix(vec),basis)
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
