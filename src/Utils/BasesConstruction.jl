function tpod(mat::Matrix,args...;ϵ=1e-4)
  tpod(Val(size(mat,1) > size(mat,2)),mat,args...;ϵ)
end

function tpod(::Val{true},mat::Matrix,args...;ϵ=1e-4)
  cmat = mat'*mat
  _,s2,V = svd(cmat)
  s = sqrt.(s2)
  n = truncation(s,ϵ)
  U = mat*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (s[i]+eps())
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

function tpod(::Val{true},mat::Matrix,X::Matrix;ϵ=1e-4)
  H = cholesky(X)
  L = sparse(H.L)
  Xmat = L'*mat[H.p,:]

  cmat = Xmat'*Xmat
  _,s2,V = svd(cmat)
  s = sqrt.(s2)
  n = truncation(s,ϵ)
  U = Xmat*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (s[i]+eps())
  end
  (L'\U[:,1:n])[invperm(H.p),:]
end

function tpod(::Val{false},mat::Matrix,X::Matrix;ϵ=1e-4)
  H = cholesky(X)
  L = sparse(H.L)
  Xmat = L'*mat[H.p,:]

  cmat = Xmat*Xmat'
  U,s2,_ = svd(cmat)
  s = sqrt.(s2)
  n = truncation(s,ϵ)
  (L'\U[:,1:n])[invperm(H.p),:]
end

function truncation(s::Vector,ϵ::Real)
  energies = cumsum(s.^2;dims=1)
  rb_ndofs = first(findall(x->x ≥ (1-ϵ^2)*energies[end],energies))[1]
  err = sqrt(1-energies[rb_ndofs]/energies[end])
  println("POD truncated at ϵ = $ϵ: number basis vectors = $rb_ndofs; projection error ≤ $err")
  rb_ndofs
end

function orth_projection(
  v::Vector{T},
  basis::Matrix{T};
  X=nothing) where T

  proj = similar(v)
  proj .= zero(T)
  @inbounds for b = eachcol(basis)
    proj += isnothing(X) ? b*sum(v'*b)/sum(b'*b) : b*sum(v'*X*b)/sum(b'*X*b)
  end
  proj
end

function orth_complement!(
  v::Vector{T},
  basis::Matrix{T};
  kwargs...) where T

  copyto!(v,v-orth_projection(v,basis;kwargs...))
end

function gram_schmidt!(
  mat::Matrix{T},
  basis::Matrix{T};
  kwargs...) where T

  @inbounds for i = axes(mat,2)
    mat_i = mat[:,i]
    orth_complement!(mat_i,basis;kwargs...)
    if i > 1
      orth_complement!(mat_i,mat[:,1:i-1];kwargs...)
    end
    mat_i /= norm(mat_i)
    mat[:,i] .= mat_i
  end
end
