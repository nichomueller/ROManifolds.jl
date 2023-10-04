function tpod(mat::AbstractMatrix,args...;ϵ=1e-4)
  tpod(Val(size(mat,1) > size(mat,2)),mat,args...;ϵ)
end

function tpod(::Val{true},mat::AbstractMatrix,args...;ϵ=1e-4)
  compressed_mat = mat'*mat
  _,Σ2,V = svd(compressed_mat)
  Σ = sqrt.(Σ2)
  n = truncation(Σ,ϵ)
  U = mat*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (Σ[i]+eps())
  end
  U
end

function tpod(::Val{false},mat::AbstractMatrix,args...;ϵ=1e-4)
  compressed_mat = mat*mat'
  U,Σ2,_ = svd(compressed_mat)
  Σ = sqrt.(Σ2)
  n = truncation(Σ,ϵ)
  U[:,1:n]
end

function tpod(::Val{true},mat::AbstractMatrix,X::AbstractMatrix;ϵ=1e-4)
  H = cholesky(X)
  L = sparse(H.L)
  Xmat = L'*mat[H.p,:]

  compressed_mat = Xmat'*Xmat
  _,Σ2,V = svd(compressed_mat)
  Σ = sqrt.(Σ2)
  n = truncation(Σ,ϵ)
  U = Xmat*V[:,1:n]
  for i = axes(U,2)
    U[:,i] /= (Σ[i]+eps())
  end
  (L'\U[:,1:n])[invperm(H.p),:]
end

function tpod(::Val{false},mat::AbstractMatrix,X::AbstractMatrix;ϵ=1e-4)
  H = cholesky(X)
  L = sparse(H.L)
  Xmat = L'*mat[H.p,:]

  compressed_mat = Xmat*Xmat'
  U,Σ2,_ = svd(compressed_mat)
  Σ = sqrt.(Σ2)
  n = truncation(Σ,ϵ)
  (L'\U[:,1:n])[invperm(H.p),:]
end

function truncation(Σ::AbstractArray,ϵ::Real)
  energies = cumsum(Σ.^2;dims=1)
  rb_ndofs = first(findall(x->x ≥ (1-ϵ^2)*energies[end],energies))[1]
  err = sqrt(1-energies[rb_ndofs]/energies[end])
  println("POD truncated at ϵ = $ϵ: number basis vectors = $rb_ndofs; projection error ≤ $err")
  rb_ndofs
end

function projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v) : v*sum(vnew'*X*v)
  proj_mat = reshape(similar(vnew),:,1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v)/sum(v'*v) : v*sum(vnew'*X*v)/sum(v'*X*v)
  proj_mat = reshape(similar(vnew),:,1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_complement(
  v::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  kwargs...)

  compl = reshape(similar(v),:,1)
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
