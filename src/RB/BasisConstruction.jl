function tpod(mat::AbstractMatrix,args...;kwargs...)
  U,Σ,V = svd(mat)
  rank = truncation(Σ;kwargs...)
  U[:,1:rank]
end

function tpod(mat::AbstractMatrix,X::AbstractMatrix;kwargs...)
  C = cholesky(X)
  L = sparse(C.L)
  Xmat = L'*mat[C.p,:]
  U,Σ,V = svd(Xmat)
  rank = truncation(Σ;kwargs...)
  (L'\U[:,1:rank])[invperm(C.p),:]
end

# we are not interested in the last dimension (corresponds to the parameter)

function ttsvd(mat::AbstractArray{T,N},args...;kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  mat_k = copy(mat)
  for k = 1:N-1
    mat_k = reshape(mat_k,ranks[k]*sizes[k],:)
    U,Σ,V = svd(mat_k)
    rank = truncation(Σ;kwargs...)
    ranks[k+1] = rank
    mat_k = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    cores[k] = reshape(U[:,1:rank],ranks[k],sizes[k],rank)
  end
  return cores
end

function ttsvd(mat::AbstractArray{T,N},X::AbstractMatrix;kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  mat_k = copy(mat)
  C = cholesky(X)
  L = sparse(C.L)
  for k = 1:N-1
    if k == 1
      _mat_k = reshape(mat_k,:,prod(sizes[2:end]))
      mat_k = reshape(L'*_mat_k[C.p,:],ranks[k]*sizes[k],:)
    else
      mat_k = reshape(mat_k,ranks[k]*sizes[k],:)
    end
    U,Σ,V = svd(mat_k)
    rank = truncation(Σ;kwargs...)
    ranks[k+1] = rank
    mat_k = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    if k == 1
      cores[k] = reshape((L'\U[:,1:rank])[invperm(C.p),:],ranks[k],sizes[k],rank)
    else
      cores[k] = reshape(U[:,1:rank],ranks[k],sizes[k],rank)
    end
  end
  return cores
end

function truncation(s::AbstractVector;ϵ=1e-4,rank=length(s))
  energies = cumsum(s.^2;dims=1)
  tolrank = findfirst(energies .>= (1-ϵ^2)*energies[end])
  min(rank,tolrank)
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
