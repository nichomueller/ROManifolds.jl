get_Nt(S::AbstractMatrix,ns::Int) = Int(size(S,2)/ns)

function mode2_unfolding(S::AbstractMatrix{Float},ns::Int)
  mode2 = Elemental.zeros(EMatrix{Float},get_Nt(S,ns),size(S,1)*ns)
  @inbounds for k = 1:ns
    copyto!(view(mode2,:,(k-1)*Ns+1:k*Ns),S[:,(k-1)*Nt+1:k*Nt]',ns)
  end
  return mode2
end

abstract type PODStyle end
struct DefaultPOD end
struct ReducedPOD end

POD(S::AbstractMatrix,args...;ϵ=1e-5,style=ReducedPOD()) = POD(style,S,args...;ϵ)

function POD(::DefaultPOD,S::AbstractMatrix,X::SparseMatrixCSC;ϵ=1e-5)
  H = cholesky(X)
  L = sparse(H.L)
  U,Σ,_ = svd(L'*S[H.p,:])
  n = truncation(Σ,ϵ)
  Matrix((L'\U[:,1:n])[invperm(H.p),:])
end

function POD(::DefaultPOD,S::AbstractMatrix;ϵ=1e-5)
  U,Σ,_ = svd(S)
  n = truncation(Σ,ϵ)
  U[:,1:n]
end

function POD(::ReducedPOD,S::AbstractMatrix;ϵ=1e-5)
  POD(Val{size(S,1)>size(S,2)}(),S;ϵ)
end

function POD(::Val{true},S::AbstractMatrix;ϵ=1e-5)
  C = S'*S
  _,_,V = svd(C)
  Σ = svdvals(S)
  n = truncation(Σ,ϵ)
  U = S*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (Σ[i]+eps())
  end
  U
end

function POD(::Val{false},S::AbstractMatrix;ϵ=1e-5)
  C = S*S'
  U,_ = svd(C)
  Σ = svdvals(S)
  n = truncation(Σ,ϵ)
  U[:,1:n]
end

function truncation(Σ::AbstractArray,ϵ::Real)
  energies = cumsum(Σ.^2;dims=1)
  n = first(findall(x->x ≥ (1-ϵ^2)*energies[end],energies))[1]
  err = sqrt(1-energies[n]/energies[end])
  printstyled("POD truncated at ϵ = $ϵ: number basis vectors = $n; projection error ≤ $err\n";
    color=:blue)
  n
end

function projection(vnew::AbstractVector,v::AbstractVector;X=nothing)
  isnothing(X) ? projection(vnew,v,I(length(v))) : projection(vnew,v,X)
end

function projection(vnew::AbstractArray,v::AbstractArray,X::AbstractMatrix)
  v*(vnew'*X*v)
end

function projection(vnew::AbstractArray,basis::AbstractMatrix;X=nothing)
  sum([projection(vnew,basis[:,i];X) for i=axes(basis,2)])
end

function orth_projection(vnew::AbstractArray,v::AbstractArray;X=nothing)
  isnothing(X) ? orth_projection(vnew,v,I(length(v))) : orth_projection(vnew,v,X)
end

function orth_projection(vnew::AbstractArray,v::AbstractArray,X::AbstractMatrix)
  projection(vnew,v,X)/(v'*X*v)
end

function orth_projection(vnew::AbstractArray,basis::AbstractMatrix;X=nothing)
  sum([orth_projection(vnew,basis[:,i];X) for i=axes(basis,2)])
end

function orth_complement(
  v::AbstractArray,
  basis::AbstractMatrix;
  kwargs...)

  v - orth_projection(v,basis;kwargs...)
end

function gram_schmidt(mat::AbstractMatrix,basis::AbstractMatrix;kwargs...)
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
