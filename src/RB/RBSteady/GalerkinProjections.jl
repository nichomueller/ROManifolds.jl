function galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix)

  proj_basis = basis_left'*basis
  return proj_basis
end

function galerkin_projection(
  basis_left::AbstractMatrix,
  basis::MatrixOfSparseMatrices,
  basis_right::AbstractMatrix)

  @check size(basis,1) == size(basis,2)
  nleft = size(basis_left,2)
  n = size(basis,1)
  nright = size(basis_right,2)

  proj_basis = zeros(nleft,n,nright)
  @inbounds for i = 1:n
    proj_basis[:,i,:] = basis_left'*param_getindex(basis,i)*basis_right
  end

  return proj_basis
end

function galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  norm_matrix::AbstractSparseMatrix)

  proj_basis = basis_left'*norm_matrix*basis
  return proj_basis
end

function galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  combine::Function)

  @notimplemented
end

function galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  basis_right::AbstractMatrix,
  combine::Function)

  nleft = size(basis_left,2)
  n = size(basis,2)
  nright = size(basis_right,2)

  proj_basis = zeros(T,nleft,n,nright)
  proj_basis′ = copy(proj_basis)

  @inbounds for i = 1:nleft, k = 1:n, j = 1:nright
    proj_basis[i,k,j] = sum(basis_left[:,i].*basis[:,k].*basis_right[:,j])
    proj_basis′[i,k,j] = sum(basis_left[2:end,i].*basis[2:end,k].*basis_right[1:end-1,j])
  end

  combine(proj_basis,proj_basis′)
end
