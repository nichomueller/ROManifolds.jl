function tpod(mat::AbstractMatrix,args...;kwargs...)
  cmat = mat'*mat
  _,s2,V = svd(cmat)
  s = sqrt.(s2)
  rank = truncation(s;kwargs...)
  U = mat*V[:,1:rank]
  for i = axes(U,2)
    U[:,i] /= s[i]+eps()
  end
  U
end

function tpod(mat::AbstractMatrix,X::AbstractMatrix;kwargs...)
  C = cholesky(X)
  L = sparse(C.L)
  Xmat = L'*mat[C.p,:]

  cmat = Xmat'*Xmat
  _,s2,V = svd(cmat)
  s = sqrt.(s2)
  rank = truncation(s;kwargs...)
  U = Xmat*V[:,1:rank]
  for i = axes(U,2)
    U[:,i] /= s[i]+eps()
  end
  (L'\U[:,1:rank])[invperm(C.p),:]
end

function truncation(s::AbstractVector;ϵ=1e-4,rank=nothing)
  if isnothing(rank)
    energies = cumsum(s.^2;dims=1)
    findfirst(x->x ≥ (1-ϵ^2)*energies[end],energies)
  else
    @check isa(rank,Integer)
    rank
  end
end

function orth_projection(
  v::AbstractVector{T},
  basis::AbstractMatrix{T},
  args...) where T

  proj = similar(v)
  proj .= zero(T)
  @inbounds for b = eachcol(basis)
    proj += b*sum(v'*b)/sum(b'*b)
  end
  proj
end

function orth_projection(
  v::AbstractVector{T},
  basis::AbstractMatrix{T},
  X::SparseMatrixCSC) where T

  proj = similar(v)
  proj .= zero(T)
  @inbounds for b = eachcol(basis)
    proj += b*sum(v'*X*b)/sum(b'*X*b)
  end
  proj
end

function orth_complement!(
  v::AbstractVector{T},
  basis::AbstractMatrix{T},
  args...) where T

  v .= v-orth_projection(v,basis,args...)
end

function gram_schmidt!(
  mat::AbstractMatrix{T},
  basis::AbstractMatrix{T},
  args...) where T

  @inbounds for i = axes(mat,2)
    mat_i = mat[:,i]
    orth_complement!(mat_i,basis,args...)
    if i > 1
      orth_complement!(mat_i,mat[:,1:i-1],args...)
    end
    mat_i /= norm(mat_i)
    mat[:,i] .= mat_i
  end
end
