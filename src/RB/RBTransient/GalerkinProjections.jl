function RBSteady.galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  combine::Function)

  @notimplemented
end

function RBSteady.galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  basis_right::AbstractMatrix,
  combine::Function)

  nleft = size(basis_left,2)
  n = size(basis,2)
  nright = size(basis_right,2)

  proj_basis = zeros(nleft,n,nright)
  proj_basis′ = copy(proj_basis)

  @inbounds for i = 1:nleft, k = 1:n, j = 1:nright
    proj_basis[i,k,j] = sum(basis_left[:,i].*basis[:,k].*basis_right[:,j])
    proj_basis′[i,k,j] = sum(basis_left[2:end,i].*basis[2:end,k].*basis_right[1:end-1,j])
  end

  combine(proj_basis,proj_basis′)
end
