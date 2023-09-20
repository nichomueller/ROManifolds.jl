abstract type PNonlinearOperator <: GridapType end

struct PAffineOperator <: PNonlinearOperator
  matrix::PTArray
  vector::PTArray
end

function Algebra.numerical_setup(ss::Algebra.LUSymbolicSetup,mat::PTArray)
  map(x->numerical_setup(ss,x),mat.array)
end

function Algebra.numerical_setup!(ns,mat::PTArray)
  map(numerical_setup!,ns,mat.array)
end

function Algebra.solve!(x::PTArray,::LinearSolver,op::PAffineOperator,ns)
  xcache = copy(testitem(x))
  A = op.matrix
  b = op.vector
  @assert length(A) == length(b) == length(x)
  numerical_setup!(ns,A)
  @inbounds for k in eachindex(x)
    x[k] = solve!(xcache,ns[k],b[k])
  end
  test_ptarray(x)
  ns
end

function Algebra.solve!(x::PTArray,ls::LinearSolver,op::PAffineOperator,::Nothing)
  xcache = copy(testitem(x))
  A = op.matrix
  b = op.vector
  @assert length(A) == length(b) == length(x)
  ss = symbolic_setup(ls,testitem(A))
  ns = numerical_setup(ss,A)
  @inbounds for k in eachindex(x)
    x[k] = solve!(xcache,ns[k],b[k])
  end
  test_ptarray(x)
  ns
end

struct PNewtonRaphsonCache <: GridapType
  A::PTArray
  b::PTArray
  dx::PTArray
  ns::NumericalSetup
end

# function ptsolve!(
#   x::PTArray,
#   xcache::AbstractVector,
#   nls::PNewtonRaphsonSolver,
#   op::PNonlinearOperator,
#   cache::PNewtonRaphsonCache)

#   b = cache.b
#   A = cache.A
#   @assert length(A) == length(b) == length(x)
#   dx = cache.dx
#   ns = cache.ns
#   residual!(b,op,x)
#   jacobian!(A,op,x)
#   numerical_setup!(ns,A)
#   _solve_nr!(x,A,b,dx,ns,nls,op)
#   cache
# end

# function ptsolve!(
#   x::PTArray,
#   xcache::AbstractVector,
#   nls::PNewtonRaphsonSolver,
#   op::PNonlinearOperator,
#   ::Nothing)

#   b = residual(op,x)
#   A = jacobian(op,x)
#   dx = similar(b)
#   ss = symbolic_setup(nls.ls,A)
#   ns = numerical_setup(ss,A)
#   _solve_nr!(x,A,b,dx,ns,nls,op)
#   NewtonRaphsonCache(A,b,dx,ns)
# end

function ptsolve!(
  x::PTArray,xcache::AbstractVector,A::PTArray,b::PTArray,cache::NumericalSetup)

  ns = cache
  for xk in eachindex(xcache)
    numerical_setup!(ns,A[k])
    solve!(xk,ns,b[k])
    x[k] = xk
  end
  cache
end

# function ptsolve!(
#   x::PTArray,
#   xcache::AbstractVector,
#   nls::PNewtonRaphsonSolver,
#   A::PTArray,
#   b::PTArray,
#   dx::PTArray,
#   cache::PNewtonRaphsonCache)

#   ns = cache.ns
#   for xk in eachindex(xcache)
#     numerical_setup!(ns,A[k])
#     Algebra._solve_nr!(xk,A,b,dx,ns,nls,op)
#     x[k] = xk
#   end
#   cache
# end
