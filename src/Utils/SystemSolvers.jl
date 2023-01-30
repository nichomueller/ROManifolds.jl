function newton(res::Function,jac::Function,X::FESpace;tol=1e-10,maxit=10)
  err = 1.
  x = zero(X)
  xh = get_free_dof_values(x)
  iter = 0
  while norm(err) > tol && iter < maxit
    jx,rx = jac(x),res(x,xh)
    err = jx \ rx
    xh -= err
    x = FEFunction(X,xh)
    iter += 1
    printstyled("\n err = $(norm(err)), iter = $iter";color=:red)
  end
  xh
end

function newton(res::Function,jac::Function,X::FESpace,t::Real,x0;tol=1e-10,maxit=10)
  err = 1.
  x = x0
  xh = get_free_dof_values(x)
  xh0 = get_free_dof_values(x0)
  iter = 0
  while norm(err) > tol && iter < maxit
    jtx,rtx = jac(t,x),res(t,x,xh,xh0)
    err = jtx \ rtx
    xh -= err
    x = FEFunction(X,xh[:,1])
    iter += 1
    printstyled("\n err = $(norm(err)), iter = $iter";color=:red)
  end
  xh
end

function solve_cholesky(
  A::SparseMatrixCSC{Float, Int},
  B::SparseMatrixCSC{Float, Int})

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] \ B
  x = L[invperm(H.p), :]' \ y

  x
end

function solve_cholesky(
  A::SparseMatrixCSC{Float, Int},
  B::Matrix{T}) where T

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] \ B
  x = L[invperm(H.p), :]' \ y

  Matrix{T}(x)
end

function solve_cholesky(
  A::SparseMatrixCSC{Float, Int},
  B::Vector{T}) where T

  H = cholesky(A)
  L = sparse(H.L)

  y = L[invperm(H.p), :] \ B
  x = L[invperm(H.p), :]' \ y

  Vector{T}(x)
end
