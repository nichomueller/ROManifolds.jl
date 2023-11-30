function tpod(mat::Matrix{T},args...;ϵ=1e-4) where T
  if isempty(mat)
    return zeros(T,0,1)
  end
  tpod(Val(size(mat,1) > size(mat,2)),mat,args...;ϵ)
end

function tpod(::Val{true},mat::Matrix,args...;ϵ=1e-4)
  cmat = mat'*mat
  _,s2,V = svd(cmat)
  s = sqrt.(s2)
  n = truncation(s,ϵ)
  U = mat*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= s[i]+eps()
  end
  U
end

function tpod(::Val{false},mat::Matrix,args...;ϵ=1e-4)
  cmat = mat*mat'
  U,s2,_ = svd(cmat)
  s = sqrt.(s2)
  n = truncation(s,ϵ)
  U[:,1:n]
end

function tpod(::Val{true},mat::Matrix,X::AbstractMatrix;ϵ=1e-4)
  C = cholesky(X)
  L = sparse(C.L)
  Xmat = L'*mat[C.p,:]

  cmat = Xmat'*Xmat
  _,s2,V = svd(cmat)
  s = sqrt.(s2)
  n = truncation(s,ϵ)
  U = Xmat*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= s[i]+eps()
  end
  (L'\U[:,1:n])[invperm(C.p),:]
end

function tpod(::Val{false},mat::Matrix,X::AbstractMatrix;ϵ=1e-4)
  C = cholesky(X)
  L = sparse(C.L)
  Xmat = L'*mat[C.p,:]

  cmat = Xmat*Xmat'
  U,s2,_ = svd(cmat)
  s = sqrt.(s2)
  n = truncation(s,ϵ)
  (L'\U[:,1:n])[invperm(C.p),:]
end

function truncation(s::Vector,ϵ::Real)
  energies = cumsum(s.^2;dims=1)
  first(findall(x->x ≥ (1-ϵ^2)*energies[end],energies))[1]
end

function orth_projection(
  v::Vector{T},
  basis::Matrix{T},
  args...) where T

  proj = similar(v)
  proj .= zero(T)
  @inbounds for b = eachcol(basis)
    proj += b*sum(v'*b)/sum(b'*b)
  end
  proj
end

function orth_projection(
  v::Vector{T},
  basis::Matrix{T},
  X::SparseMatrixCSC) where T

  proj = similar(v)
  proj .= zero(T)
  @inbounds for b = eachcol(basis)
    proj += b*sum(v'*X*b)/sum(b'*X*b)
  end
  proj
end

function orth_complement!(
  v::Vector{T},
  basis::Matrix{T},
  args...) where T

  v .= v-orth_projection(v,basis,args...)
end

function gram_schmidt!(
  mat::Matrix{T},
  basis::Matrix{T},
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
