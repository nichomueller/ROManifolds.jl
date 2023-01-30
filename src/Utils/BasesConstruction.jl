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

function POD(S::NTuple{N,AbstractMatrix},args...;kwargs...) where N
  Broadcasting(si -> POD(si,args...;kwargs...))(S)
end

function POD(S::AbstractMatrix,X::SparseMatrixCSC;ϵ=1e-5)
  H = cholesky(X)
  L = sparse(H.L)
  U,Σ,_ = my_svd(L'*S[H.p, :])

  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("\n Basis number obtained via POD is $n, projection error ≤ $err";
    color=:blue)

  Matrix((L'\U[:,1:n])[invperm(H.p),:])
end

POD(S::AbstractMatrix;ϵ=1e-5) = POD(S,Val(false);ϵ=ϵ)

function POD(S::AbstractMatrix,::Val{false};ϵ=1e-5)
  U,Σ,_ = my_svd(S)
  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("\n Basis number obtained via POD is $n, projection error ≤ $err"
    ;color=:blue)

  U[:,1:n]
end

function POD(S::AbstractMatrix,::Val{true};ϵ=1e-5)
  nrow,ncol = size(S)
  approx_POD(S,Val(nrow>ncol);ϵ=ϵ)
end

function approx_POD(S::AbstractMatrix,::Val{true};ϵ=1e-5)
  C = S'*S
  _,_,V = my_svd(C)
  Σ = svdvals(S)

  energies = cumsum(Σ.^2)
  ntemp = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]

  matrix_err = sqrt(ntemp)*vcat(Σ[2:end],0.0)
  n = findall(x -> x ≤ ϵ,matrix_err)[1]
  err = matrix_err[n]
  printstyled("\n Basis number obtained via approximated POD is $n,
    projection error ≤ $err";color=:blue)

  U = S*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (Σ[i]+eps())
  end
  U
end

function approx_POD(S::AbstractMatrix,::Val{false};ϵ=1e-5)
  C = S*S'
  U,_ = my_svd(C)
  Σ = svdvals(S)

  energies = cumsum(Σ)
  ntemp = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]

  matrix_err = sqrt(ntemp)*vcat(Σ[2:end],0.0)
  n = findall(x -> x ≤ ϵ,matrix_err)[1]
  err = matrix_err[n]
  printstyled("\n Basis number obtained via approximated POD is $n,
    projection error ≤ $err";color=:blue)

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
