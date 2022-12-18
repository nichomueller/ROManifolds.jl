get_Nt(S::AbstractMatrix,ns::Int) = Int(size(S,2)/ns)

function mode2_unfolding(S::AbstractMatrix,ns::Int)
  Nt = get_Nt(S,ns)
  idx_fun(ns) = (ns .- 1)*Nt .+ 1:ns*Nt
  idx = idx_fun.(1:ns)
  mode2_blocks(i) = Matrix(transpose(getindex(S,:,i)))
  mode2 = Matrix(mode2_blocks.(idx))

  mode2
end

my_svd(s::Matrix{Float}) = svd(s)
my_svd(s::SparseMatrixCSC) = svds(s;nsv=size(s)[2]-1)[1]
my_svd(s::Vector{AbstractMatrix}) = my_svd(Matrix(s))

function POD(S::AbstractMatrix,X::SparseMatrixCSC;ϵ=1e-5)
  H = cholesky(X)
  L = sparse(H.L)
  U,Σ,_ = my_svd(L'*S[H.p, :])

  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  println("Basis number obtained via POD is $n, projection error ≤ $err")

  Matrix((L'\U[:,1:n])[invperm(H.p),:])
end

POD(S::AbstractMatrix;ϵ=1e-5) = POD(S,Val(false);ϵ=ϵ)

function POD(S::AbstractMatrix,::Val{false};ϵ=1e-5)
  U,Σ,_ = my_svd(S)
  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  println("Basis number obtained via POD is $n, projection error ≤ $err")

  U[:,1:n]
end

function POD(S::AbstractMatrix,::Val{true};ϵ=1e-5)
  U,Σ,_ = my_svd(S)
  energies = cumsum(Σ.^2)
  ntemp = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  Utemp = U[:,1:ntemp]

  # approx, should actually be norm(inv(U[idx,:]))*Σ[2:end]
  matrix_err = sqrt(norm(inv(Utemp'*Utemp)))*vcat(Σ[2:end],0.0)
  n = findall(x -> x ≤ ϵ,matrix_err)[1]
  err = matrix_err[n]
  println("Basis number obtained via POD is $n, projection error ≤ $err")

  U[:,1:n]
end

function projection(vnew::AbstractVector,v::AbstractVector)
  v*(vnew'*v)
end

function projection(vnew::AbstractVector,basis::AbstractMatrix)
  sum([projection(vnew,basis[:,i]) for i=axes(basis,2)])
end

function orth_projection(vnew::AbstractVector,v::AbstractVector)
  projection(vnew,v)/(v'*v)
end

function orth_projection(vnew::AbstractVector,basis::AbstractMatrix)
  sum([orth_projection(vnew,basis[:,i]) for i=axes(basis,2)])
end

isbasis(basis,args...) =
  all([isapprox(norm(basis[:,j],args...),1) for j=axes(basis,2)])

function orth_complement(
  v::AbstractVector,
  basis::AbstractMatrix)

  v - projection(v,basis)
end

function gram_schmidt(mat::Matrix{Float},basis::Matrix{Float})

  for i = axes(mat,2)
    mat_i = mat[:,i]
    mat_i = orth_complement(mat_i,basis)
    if i > 1
      mat_i = orth_complement(mat_i,mat[:,1:i-1])
    end
    mat[:,i] = mat_i/norm(mat_i)
  end

  mat
end

function gram_schmidt(vec::Vector{Vector},basis::Matrix{Float})
  gram_schmidt(Matrix(vec),basis)
end

function basis_by_coeff_mult(basis::Matrix{Float},coeff::Vector{Float},nr::Int)
  @assert size(basis,2) == length(coeff) "Something is wrong"
  bc = sum([basis[:,k]*coeff[k] for k=eachindex(coeff)])
  Matrix(reshape(bc,nr,:))
end

function basis_by_coeff_mult(
  basis::NTuple{2,Matrix{Float}},
  coeff::NTuple{2,Vector{Float}},
  nr::Int)

  Broadcasting((b,c)->basis_by_coeff_mult(b,c,nr))(basis,coeff)
end

function basis_by_coeff_mult(basis::Vector{Matrix{Float}},coeff::Vector{Matrix{Float}},nr::Int)
  @assert length(basis) == length(coeff) "Something is wrong"
  bc = sum([kron(basis[k],coeff[k]) for k=eachindex(coeff)])
  Matrix(reshape(bc,nr,:))
end

function basis_by_coeff_mult(
  basis::T,
  coeff::NTuple{2,T},
  nr::Int) where T

  Broadcasting(c->basis_by_coeff_mult(basis,c,nr))(coeff)
end

function basis_by_coeff_mult(
  basis::NTuple{2,T},
  coeff::Tuple{NTuple{2,T},T},
  nr::Int) where T

  matinfo = first(basis),first(coeff)
  liftinfo = last(basis),last(coeff)
  basis_by_coeff_mult(matinfo...,nr),basis_by_coeff_mult(liftinfo...,nr)
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

function Base.Matrix(v::Vector{T}) where T
  Matrix{T}(reshape(v,:,1))
end

function Base.Matrix(vblock::Vector{Vector{T}}) where T
  Matrix{T}(reduce(vcat,transpose.(vblock))')
end

#= function Base.Matrix(mblock::Vector{T}) where {T<:AbstractMatrix}
  @assert check_dimensions(mblock) "Wrong dimensions"
  T(reduce(vcat,transpose.(mblock))')
end =#
function Base.Matrix(mblock::Vector{T}) where {T<:AbstractMatrix}
  @assert check_dimensions(mblock) "Wrong dimensions"
  T(reduce(hcat,mblock))
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

function blocks(mat::NTuple{2,Matrix{T}},nrow::Int) where T
  Broadcasting(m -> blocks(m,nrow))(mat)
end

function blocks(mat::Matrix{T},nrow::Int) where T
  blocks(mat,size(mat,2);dims=(nrow,:))
end

function blocks(mat::Matrix{T},nblocks=1;dims=(size(mat,1),1)) where T
  @assert check_dimensions(mat,nblocks) "Wrong dimensions"

  ncol_block = Int(size(mat)[2]/nblocks)
  idx2 = ncol_block:ncol_block:size(mat)[2]
  idx1 = idx2 .- ncol_block .+ 1

  blockmat = Matrix{T}[]
  for i in eachindex(idx1)
    block_i = Matrix(reshape(mat[:,idx1[i]:idx2[i]],dims))
    push!(blockmat,block_i)
  end
  blockmat::Vector{Matrix{T}}
end

function blocks(mat::Array{T,3};dims=size(mat)[1:2]) where T
  blockmat = Matrix{T}[]
  for nb = 1:size(mat,3)
    block_i = Matrix(reshape(mat[:,:,nb],dims))
    push!(blockmat,block_i)
  end
  blockmat::Vector{Matrix{T}}
end

function vblocks(mat::Matrix{T}) where T
  blockvec = Vector{T}[]
  for i in axes(mat,2)
    push!(blockvec,mat[:,i])
  end
  blockvec::Vector{Vector{T}}
end

check_dimensions(vb::AbstractVector) =
  all([size(vb[i])[1] == size(vb[1])[1] for i = 2:length(vb)])
check_dimensions(m::AbstractMatrix,nb::Int) = iszero(size(m)[2]%nb)

#= function remove_zero_rows(m::AbstractMatrix)
  sum_row = sum(abs.(m),dims=2)[:,1]
  significant_rows = findall(x -> x > eps(),sum_row)
  m[significant_rows,:]
end =#

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

Base.:(^)(vv::VectorValue,n::Int) = VectorValue([vv[k]^n for k=eachindex(vv)])
Base.:(^)(vv::VectorValue,n::Float) = VectorValue([vv[k]^n for k=eachindex(vv)])

Gridap.get_triangulation(m::Measure) = m.quad.trian
