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
  mode2_blocks = Matrix.(transpose.(S[idx]))
  mode2 = Matrix(mode2_blocks)

  mode2
end

my_svd(s::Matrix) = svd(s)
my_svd(s::SparseMatrixCSC) = svds(s;nsv=size(S)[2]-1)[1]
my_svd(s::Vector{AbstractMatrix}) = my_svd(Matrix(s))

function POD(S::AbstractMatrix,ϵ=1e-5)
  U,Σ,_ = my_svd(S)
  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  println("Basis number obtained via POD is $n, projection error ≤ $err")

  U[:,1:n]
end

function POD(S::AbstractMatrix,X::SparseMatrixCSC,ϵ=1e-5)
  H = cholesky(X)
  L = sparse(H.L)
  U,Σ,_ = my_svd(L'*S[H.p, :])

  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  println("Basis number obtained via POD is $n, projection error ≤ $err")

  Matrix((L'\U[:,1:n])[invperm(H.p),:])
end

function POD_for_MDEIM(S::AbstractMatrix,ϵ=1e-5)
  U,Σ,_ = my_svd(S)
  energies = cumsum(Σ.^2)
  ntemp = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  Utemp = U[:,1:ntemp]

  # approx, should actually be norm(inv(U[idx,:]))*Σ[2:end]
  matrix_err = vcat(sqrt(norm(inv(Utemp'*Utemp)))*Σ[2:end],0.0)
  n = findall(x -> x ≤ ϵ,matrix_err)[1]
  err = matrix_err[n]
  println("Basis number obtained via POD is $n, projection error ≤ $err")

  U[:,1:n]
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

function gram_schmidt!(mat::Matrix,basis::Matrix)

  println("Normalizing primal supremizer 1")
  mat[:,1] = orth_complement(mat[:,1],basis)

  for i = 2:size(mat,2)
    println("Normalizing primal supremizer $i")
    mat[:,i] = orth_complement(mat[:,i],basis)
    mat[:,i] = orth_complement(mat[:,i],vec[:,1:i-1])
    supr_norm = norm(mat[:,i])
    println("Norm supremizers: $supr_norm")
    mat[:,i] /= supr_norm
  end

  mat
end

function gram_schmidt!(vec::Vector{Vector},basis::Matrix)
  gram_schmidt!(Matrix(vec),basis)
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

function Base.Matrix(vblock::Vector{Vector{T}}) where T
  Matrix{T}(reduce(vcat,transpose.(vblock))')
end

function Base.Matrix(mblock::Vector{T}) where {T<:AbstractMatrix}
  @assert check_dimensions(mblock) "Wrong dimensions"
  T(reduce(vcat,transpose.(mblock))')
end

function Base.Matrix(vblock::Vector{Vector{Vector{T}}}) where T
  n = length(vblock)
  mat = Matrix(vblock[1])
  if n > 1
    for i = 2:n
      mat = hcat(mat,Matrix(vblock[i]))
    end
  end
  mat
end

function blocks(m::Matrix{T},nblocks=1) where T
  @assert check_dimensions(m,nblocks) "Wrong dimensions"

  ncol_block = Int(size(m)[2]/nblocks)
  idx2 = ncol_block:ncol_block:size(m)[2]
  idx1 = idx2 .- ncol_block .+ 1

  blockmat = Matrix{T}[]
  for i in eachindex(idx1)
    push!(blockmat,m[:,idx1[i]:idx2[i]])
  end
  blockmat::Vector{Matrix{T}}
end

function blocks(mat::Array{T,3}) where T
  blockmat = Matrix{T}[]
  for nb = 1:size(mat)[end]
    push!(blockmat,mat[:,:,nb])
  end
  blockmat
end

function vblocks(m::Matrix{T}) where T
  blockvec = Vector{T}[]
  for i in axes(m,2)
    push!(blockvec,m[:,i])
  end
  blockvec::Vector{Vector{T}}
end

check_dimensions(vb::AbstractVector) =
  all([size(vb[i])[1] == size(vb[1])[1] for i = 2:length(vb)])
check_dimensions(m::AbstractMatrix,nb::Int) = iszero(size(m)[2]%nb)

#= function SparseArrays.findnz(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
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
end =#

function Base.NTuple(N::Int,T::DataType)
  NT = ()
  for _ = 1:N
    NT = (NT...,zero(T))
  end
  NT::NTuple{N,T}
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

Base.:(*)(a::Symbol,b::Symbol) = Symbol(String(a)*String(b))

Gridap.get_triangulation(m::Measure) = m.quad.trian
