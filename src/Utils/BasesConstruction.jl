get_Nt(S::AbstractMatrix,ns::Int) = Int(size(S,2)/ns)

function mode2_unfolding(S::DistMatrix{Float},ns::Int)
  mode2 = mode2_unfolding(Matrix(S),ns)
  DistMatrix(mode2)
end

function mode2_unfolding(S::Matrix{Float},ns::Int)
  mode2 = Matrix{Float}(undef,get_Nt(S,ns),size(S,1)*ns)
  @inbounds for k = 1:ns
    copyto!(view(mode2,:,(k-1)*Ns+1:k*Ns),S[:,(k-1)*Nt+1:k*Nt]',ns)
  end
  return mode2
end

function POD(S::AbstractMatrix,X::SparseMatrixCSC;ϵ=1e-5)
  H = cholesky(X)
  L = sparse(H.L)
  U,Σ,_ = svd(L'*S[H.p,:])
  n = truncation(Σ,ϵ)
  Matrix((L'\U[:,1:n])[invperm(H.p),:])
end

function POD(S::AbstractMatrix;ϵ=1e-5)
  U,Σ,_ = svd(S)
  n = truncation(Σ,ϵ)
  U[:,1:n]
end

function reduced_POD(S::AbstractMatrix;ϵ=1e-5)
  reduced_POD(Val{size(S,1)>size(S,2)}(),S;ϵ)
end

function reduced_POD(::Val{true},S::AbstractMatrix;ϵ=1e-5)
  C = S'*S
  _,_,V = svd(C)
  Σ = svdvals(S)
  n = truncation(Σ,ϵ)
  correct_basis(S,V[:,1:n],Σ)
end

function reduced_POD(::Val{false},S::AbstractMatrix;ϵ=1e-5)
  C = S*S'
  U,_ = svd(C)
  Σ = svdvals(S)
  n = truncation(Σ,ϵ)
  U[:,1:n]
end

function truncation(Σ::Vector{Float},ϵ::Real)
  energies = cumsum(Σ.^2)
  n = findall(x->x ≥ (1-ϵ^2)*energies[end],energies)[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("Basis number obtained via POD is $n, projection error ≤ $err\n";
    color=:blue)
  n
end

truncation(Σ::DistMatrix{Float},ϵ::Real) = truncation(Matrix(Σ)[:],ϵ)

function correct_basis(S::Matrix{Float},V::Matrix{Float},Σ::Vector{Float})
  U = S*V
  for i = axes(U,2)
    U[:,i] /= (Σ[i]+eps())
  end
  U
end

function correct_basis(S::DistMatrix{Float},V::DistMatrix{Float},Σ::DistMatrix{Float})
  basis = correct_basis(Matrix(S),Matrix(V),Matrix(Σ)[:])
  DistMatrix(basis)
end

function projection(vnew::AbstractVector,v::AbstractVector;X=nothing)
  isnothing(X) ? projection(vnew,v,I(length(v))) : projection(vnew,v,X)
end

function projection(vnew::AbstractVector,v::AbstractVector,X::AbstractMatrix)
  v*(vnew'*X*v)
end

function projection(vnew::AbstractVector,basis::AbstractMatrix;X=nothing)
  sum([projection(vnew,basis[:,i];X) for i=axes(basis,2)])
end

function orth_projection(vnew::AbstractVector,v::AbstractVector;X=nothing)
  isnothing(X) ? orth_projection(vnew,v,I(length(v))) : orth_projection(vnew,v,X)
end

function orth_projection(vnew::AbstractVector,v::AbstractVector,X::AbstractMatrix)
  projection(vnew,v,X)/(v'*X*v)
end

function orth_projection(vnew::AbstractVector,basis::AbstractMatrix;X=nothing)
  sum([orth_projection(vnew,basis[:,i];X) for i=axes(basis,2)])
end

function orth_complement(
  v::AbstractVector,
  basis::AbstractMatrix;
  kwargs...)

  v - orth_projection(v,basis;kwargs...)
end

function gram_schmidt(mat::Matrix{Float},basis::Matrix{Float};kwargs...)
  for i = axes(mat,2)
    mat_i = mat[:,i]
    mat_i = orth_complement(mat_i,basis;kwargs...)
    if i > 1
      mat_i = orth_complement(mat_i,mat[:,1:i-1];kwargs...)
    end
    mat[:,i] = mat_i/norm(mat_i)
  end

  mat
end
