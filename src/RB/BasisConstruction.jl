function tpod(mat::AbstractMatrix,args...;nested=false,kwargs...)
  if nested
    _nested_tpod(mat,args...;kwargs...)
  else
    _tpod(mat,args...;kwargs...)
  end
end

function _tpod(mat::AbstractMatrix,args...;kwargs...)
  U,Σ,V = svd(mat)
  rank = truncation(Σ;kwargs...)
  U[:,1:rank]
end

function _tpod(mat::AbstractMatrix,X::AbstractMatrix;kwargs...)
  C = cholesky(X)
  L = sparse(C.L)
  Xmat = L'*mat[C.p,:]
  U,Σ,V = svd(Xmat)
  rank = truncation(Σ;kwargs...)
  (L'\U[:,1:rank])[invperm(C.p),:]
end

function _nested_tpod(mat::AbstractMatrix,args...;kwargs...)
  U,Σ,V = svd(mat)
  rank = truncation(Σ;kwargs...)
  U_rank = U[:,1:rank]
  for i = axes(V,2)
    V[:,i] .*= Σ[i]
  end
  ΣV_rank = ΣV[:,1:rank]
  UΣV_rank, = svd(ΣV_rank)
  U_rank*UΣV_rank
end

function _nested_tpod(mat::AbstractMatrix,X::AbstractMatrix;kwargs...)
  C = cholesky(X)
  L = sparse(C.L)
  Xmat = L'*mat[C.p,:]
  U,Σ,V = svd(Xmat)
  rank = truncation(Σ;kwargs...)
  U_rank = U[:,1:rank]
  for i = axes(V,2)
    V[:,i] .*= Σ[i]
  end
  ΣV_rank = ΣV[:,1:rank]
  UΣV_rank, = svd(ΣV_rank)
  (L'\U_rank*UΣV_rank)[invperm(C.p),:]
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
  v::AbstractVector,
  basis::AbstractMatrix)

  proj = similar(v)
  proj .= zero(eltype(proj))
  @inbounds for b = eachcol(basis)
    proj += b*dot(v,b)/dot(b,b)
  end
  proj
end

function orth_projection(
  v::AbstractVector,
  basis::AbstractMatrix,
  X::AbstractMatrix)

  proj = similar(v)
  proj .= zero(eltype(proj))
  @inbounds for b = eachcol(basis)
    proj += b*dot(v,X,b)/dot(b,X,b)
  end
  proj
end

function orth_complement!(
  v::AbstractVector,
  basis::AbstractMatrix,
  args...)

  v .= v-orth_projection(v,basis,args...)
end

_norm(v::AbstractVector,args...) = norm(v)
_norm(v::AbstractVector,X::AbstractMatrix) = sqrt(v'*X*v)

function gram_schmidt!(
  mat::AbstractMatrix,
  basis::AbstractMatrix,
  args...)

  @inbounds for i = axes(mat,2)
    mat_i = mat[:,i]
    orth_complement!(mat_i,basis,args...)
    if i > 1
      orth_complement!(mat_i,mat[:,1:i-1],args...)
    end
    mat_i /= _norm(mat_i,args...)
    mat[:,i] .= mat_i
  end
end
