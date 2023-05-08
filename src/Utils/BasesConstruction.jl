get_Nt(S::AbstractMatrix,ns::Int) = Int(size(S,2)/ns)

function mode2_unfolding(S::AbstractMatrix{Float},ns::Int)
  Ns,Nt = size(S,1),get_Nt(S,ns)
  mode2 = allocate_matrix(S,Nt,Ns*ns)
  @inbounds for k = 1:ns
    copyto!(view(mode2,:,(k-1)*Ns+1:k*Ns),S[:,(k-1)*Nt+1:k*Nt]')
  end
  return mode2
end

abstract type PODStyle end
struct DefaultPOD end
struct ReducedPOD end

POD(S::AbstractMatrix,args...;ϵ=1e-4,style=ReducedPOD()) = POD(style,S,args...;ϵ)

function POD(::DefaultPOD,S::AbstractMatrix,X::SparseMatrixCSC;ϵ=1e-4)
  H = cholesky(X)
  L = sparse(H.L)
  U,Σ,_ = svd(L'*S[H.p,:])
  n = truncation(Σ,ϵ)
  (L'\U[:,1:n])[invperm(H.p),:]
end

function POD(::DefaultPOD,S::AbstractMatrix;ϵ=1e-4)
  U,Σ,_ = svd(S)
  n = truncation(Σ,ϵ)
  U[:,1:n]
end

function POD(::ReducedPOD,S::AbstractMatrix;ϵ=1e-4)
  POD(Val{size(S,1)>size(S,2)}(),S;ϵ)
end

function POD(::Val{true},S::AbstractMatrix;ϵ=1e-4)
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

function POD(::Val{false},S::AbstractMatrix;ϵ=1e-4)
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

function projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v) : v*sum(vnew'*X*v)
  proj_mat = allocate_matrix(basis,length(vnew),1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v)/sum(v'*v) : v*sum(vnew'*X*v)/sum(v'*X*v)
  proj_mat = allocate_matrix(basis,length(vnew),1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_complement(
  v::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  kwargs...)

  compl = allocate_matrix(basis,length(v),1)
  copyto!(compl,v - orth_projection(v,basis;kwargs...))
end

function gram_schmidt(
  mat::AbstractMatrix{Float},
  basis::AbstractMatrix{Float};
  kwargs...)

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
