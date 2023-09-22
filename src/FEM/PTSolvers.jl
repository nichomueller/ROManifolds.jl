abstract type PNonlinearOperator <: NonlinearOperator end

struct PAffineOperator <: PNonlinearOperator
  matrix::PTArray
  vector::PTArray
end

function Algebra.numerical_setup(ss::Algebra.LUSymbolicSetup,mat::PTArray,args...)
  ns = Vector{LUNumericalSetup}(undef,length(mat))
  @inbounds for k = eachindex(mat)
    ns[k] = numerical_setup(ss,mat[k])
  end
  ns
end

function Algebra.numerical_setup(::Algebra.LUSymbolicSetup,mat::PTArray,::Affine)
  lu1 = lu(mat[1])
  ns = Vector{LUNumericalSetup}(undef,length(mat))
  @inbounds for k = eachindex(mat)
    ns[k] = LUNumericalSetup(copy(lu1))
  end
  ns
end

function Algebra.numerical_setup!(ns,mat::PTArray,args...)
  @inbounds for k = eachindex(mat)
    ns[k].factors = lu(mat[k])
  end
end

function Algebra.numerical_setup!(ns,mat::PTArray,::Affine)
  lu1 = lu(mat[1])
  @inbounds for k = eachindex(mat)
    ns[k].factors = copy(lu1)
  end
end

function _loop_solve!(x::PTArray,ns,b::PTArray,::Affine,::Affine)
  x1 = copy(x1)
  solve!(x1,ns[1],b[1])
  @inbounds for k in eachindex(x)
    x[k] = copy(x1)
  end
end

function _loop_solve!(x::PTArray,ns,b::PTArray,args...)
  @inbounds for k in eachindex(x)
    solve!(x[k],ns[k],b[k])
  end
end

function Algebra.solve!(x::PTArray,ls::LinearSolver,op::PAffineOperator,::Nothing)
  A,b = op.matrix,op.vector
  @assert length(A) == length(b) == length(x)
  Aaff,baff = get_affinity(A.array),get_affinity(b.array)
  ss = symbolic_setup(ls,testitem(A))
  ns = numerical_setup(ss,A,Aaff)
  _loop_solve!(x,ns,b,Aaff,baff)
  ns
end

function Algebra.solve!(x::PTArray,::LinearSolver,op::PAffineOperator,ns)
  A,b = op.matrix,op.vector
  Aaff,baff = get_affinity(A.array),get_affinity(b.array)
  numerical_setup!(ns,A,Aaff)
  _loop_solve!(x,ns,b,Aaff,baff)
  ns
end

struct PNewtonRaphsonCache <: GridapType
  A::PTArray
  b::PTArray
  dx::PTArray
  ns::Vector{<:NumericalSetup}
end

function Algebra.solve!(
  x::PTArray,
  nls::NewtonRaphsonSolver,
  op::PNonlinearOperator,
  ::Nothing)

  b = residual(op,x)
  A = jacobian(op,x)
  dx = similar(b)
  @assert length(A) == length(b) == length(x)
  Aaff,baff = get_affinity(A.array),get_affinity(b.array)
  ss = symbolic_setup(nls.ls,testitem(A))
  ns = numerical_setup(ss,A,Aaff)
  Algebra._solve_nr!(x,A,b,dx,ns,nls,op,Aaff,baff)
  PNewtonRaphsonCache(A,b,dx,ns)
end

function Algebra.solve!(
  x::PTArray,
  nls::NewtonRaphsonSolver,
  op::PNonlinearOperator,
  cache::PNewtonRaphsonCache)

  b = cache.b
  A = cache.A
  dx = cache.dx
  ns = cache.ns
  residual!(b,op,x)
  jacobian!(A,op,x)
  Aaff,baff = get_affinity(A.array),get_affinity(b.array)
  numerical_setup!(ns,A,Aaff)
  Algebra._solve_nr!(x,A,b,dx,ns,nls,op,Aaff,baff)
  cache
end

function Algebra._solve_nr!(x,A,b,dx,ns,nls,op,Aaff,baff)
  isconv,conv0 = Algebra._check_convergence(nls,b)
  if isconv; return; end
  for iter in 1:nls.max_nliters
    b.array .*= -1
    _loop_solve!(dx,ns,b,Aaff,baff)
    x.array .+= dx.array
    residual!(b,op,x)
    isconv = Algebra._check_convergence(nls,b,conv0)
    if isconv; return; end
    if iter == nls.max_nliters
      @unreachable
    end
    jacobian!(A,op,x)
    numerical_setup!(ns,A,Aaff)
  end
end
