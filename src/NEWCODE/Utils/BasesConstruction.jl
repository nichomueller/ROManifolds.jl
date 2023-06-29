_compress_rows(mat::AbstractMatrix) = size(mat,1) > size(mat,2)

function tpod(mat::AbstractMatrix;ϵ=1e-4)
  by_row = _compress_rows(mat)
  tpod(Val{by_row}(),mat;ϵ)
end

function tpod(::Val{true},mat::AbstractMatrix;ϵ=1e-4)
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

function tpod(::Val{false},mat::AbstractMatrix;ϵ=1e-4)
  compressed_mat = mat*mat'
  U,Σ2,_ = svd(compressed_mat)
  Σ = sqrt.(Σ2)
  n = truncation(Σ,ϵ)
  U[:,1:n]
end

function truncation(Σ::AbstractArray,ϵ::Real)
  energies = cumsum(Σ.^2;dims=1)
  rb_ndofs = first(findall(x->x ≥ (1-ϵ^2)*energies[end],energies))[1]
  err = sqrt(1-energies[rb_ndofs]/energies[end])
  printstyled("POD truncated at ϵ = $ϵ: number basis vectors = $rb_ndofs; projection error ≤ $err\n";
    color=:blue)
  rb_ndofs
end

function change_mode(mat::AbstractMatrix,nparams::Int)
  mode1_ndofs = size(mat,1)
  mode2_ndofs = Int(size(mat,2)/nparams)

  mode2 = reshape(mat,mode2_ndofs,mode1_ndofs*nparams)
  _mode2(k::Int) = mat[:,(k-1)*mode2_ndofs+1:k*mode2_ndofs]'
  @inbounds for k = 1:nparams
    setindex!(mode2,_mode2(k),:,(k-1)*mode1_ndofs+1:k*mode1_ndofs)
  end

  mode2
end

function projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v) : v*sum(vnew'*X*v)
  proj_mat = reshape(similar(basis),:,1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_projection(
  vnew::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  X=nothing)

  proj(v) = isnothing(X) ? v*sum(vnew'*v)/sum(v'*v) : v*sum(vnew'*X*v)/sum(v'*X*v)
  proj_mat = reshape(similar(basis),:,1)
  copyto!(proj_mat,sum([proj(basis[:,i]) for i = axes(basis,2)]))
  proj_mat
end

function orth_complement(
  v::AbstractArray{Float},
  basis::AbstractMatrix{Float};
  kwargs...)

  compl = reshape(similar(basis),:,1)
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
