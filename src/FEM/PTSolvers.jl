function numerical_setup(ss::Algebra.LUSymbolicSetup,mat::NonaffinePTArray)
  ns = Vector{LUNumericalSetup}(undef,length(mat))
  @inbounds for k = eachindex(mat)
    ns[k] = numerical_setup(ss,mat[k])
  end
  ns
end

function numerical_setup(ss::Algebra.LUSymbolicSetup,mat::AffinePTArray)
  ns = Vector{LUNumericalSetup}(undef,1)
  ns[1] = numerical_setup(ss,mat[1])
  ns
end

function numerical_setup!(ns,mat::NonaffinePTArray)
  @inbounds for k = eachindex(mat)
    ns[k].factors = lu(mat[k])
  end
end

function numerical_setup!(ns,mat::AffinePTArray)
  ns[1].factors = lu(mat[1])
end

function _loop_solve!(x::PTArray,ns,b::PTArray)
  @inbounds for k in eachindex(x)
    solve!(x[k],ns[k],b[k])
  end
end

struct PTAffineOperator <: PTOperator{Affine}
  matrix::PTArray
  vector::PTArray
end

function solve!(x::PTArray,ls::LinearSolver,op::PTAffineOperator,::Nothing)
  A,b = op.matrix,op.vector
  @assert length(A) == length(b) == length(x)
  ss = symbolic_setup(ls,testitem(A))
  ns = numerical_setup(ss,A)
  _loop_solve!(x,ns,b)
  ns
end

function solve!(x::PTArray,::LinearSolver,op::PTAffineOperator,ns)
  A,b = op.matrix,op.vector
  numerical_setup!(ns,A)
  _loop_solve!(x,ns,b)
  ns
end

struct PNewtonRaphsonCache <: GridapType
  A::PTArray
  b::PTArray
  dx::PTArray
  ns::Vector{<:NumericalSetup}
end

function _inf_norm(b::AbstractArray)
  m = 0
  for bi in b
    m = max(m,abs(bi))
  end
  m
end

function Algebra._check_convergence(tol,b::PTArray)
  n = length(b)
  m0 = map(_inf_norm,b.array)
  ntuple(i->false,Val(n)),m0
end

function Algebra._check_convergence(tol,b::PTArray,m0)
  m = map(_inf_norm,b.array)
  m .< tol * m0,m
end

function solve!(
  x::PTArray,
  nls::NewtonRaphsonSolver,
  op::PTOperator,
  ::Nothing)

  b = allocate_residual(op,x)
  residual!(b,op,x)
  A = allocate_jacobian(op,x)
  jacobian!(A,op,x)
  dx = similar(b)
  @assert length(A) == length(b) == length(x)
  ss = symbolic_setup(nls.ls,testitem(A))
  ns = numerical_setup(ss,A)
  Algebra._solve_nr!(x,A,b,dx,ns,nls,op)
  PNewtonRaphsonCache(A,b,dx,ns)
end

function solve!(
  x::PTArray,
  nls::NewtonRaphsonSolver,
  op::PTOperator,
  cache::PNewtonRaphsonCache)

  b = cache.b
  A = cache.A
  dx = cache.dx
  ns = cache.ns
  residual!(b,op,x)
  jacobian!(A,op,x)
  numerical_setup!(ns,A)
  Algebra._solve_nr!(x,A,b,dx,ns,nls,op)
  cache
end

function Algebra._solve_nr!(x::PTArray,A::PTArray,b::PTArray,dx::PTArray,ns,nls,op)
  _,conv0 = Algebra._check_convergence(nls.tol,b)
  for iter in 1:nls.max_nliters
    b.array .*= -1
    _loop_solve!(dx,ns,b)
    x .+= dx
    residual!(b,op,x)
    isconv,conv = Algebra._check_convergence(nls.tol,b,conv0)
    println("Iter $iter, f(x;μ) inf-norm ∈ $((minimum(conv),maximum(conv)))")
    if all(isconv); return; end
    if iter == nls.max_nliters
      @unreachable
    end
    jacobian!(A,op,x)
    numerical_setup!(ns,A)
  end
end
