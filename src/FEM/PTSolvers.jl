abstract type PNonlinearOperator <: GridapType end

struct PAffineOperator <: PNonlinearOperator
  matrix::PTArray
  vector::PTArray
end

function Algebra.solve!(x::PTArray,solver,op,args...)
  xcache = zeros(x)
  ptsolve!(x,xcache,solver,op,args...)
end

function Algebra.numerical_setup(ss::Algebra.LUSymbolicSetup,mat::PTArray)
  map(x->numerical_setup(ss,x),mat.array)
end

function Algebra.numerical_setup!(ns,mat::PTArray)
  map(numerical_setup!,ns,mat.array)
end

function ptsolve!(
  x::PTArray,xcache::AbstractVector,::LinearSolver,op::PAffineOperator,ns)

  A = op.matrix
  b = op.vector
  @assert length(A) == length(b) == length(x)
  numerical_setup!(ns,A)
  for (k,xk) in enumerate(xcache)
    solve!(xk,ns[k],b[k])
    x.array[k] = xk
  end
  ns
end

function ptsolve!(
  x::PTArray,xcache::AbstractVector,ls::LinearSolver,op::PAffineOperator,::Nothing)

  A = op.matrix
  b = op.vector
  @assert length(A) == length(b) == length(x)
  ss = symbolic_setup(ls,testitem(A))
  ns = numerical_setup(ss,A)
  for (k,xk) in enumerate(xcache)
    solve!(xk,ns[k],b[k])
    x.array[k] = xk
  end
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
    x.array[k] = xk
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
#     x.array[k] = xk
#   end
#   cache
# end
