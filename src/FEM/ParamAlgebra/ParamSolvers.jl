# linear interface

function Algebra.solve(
  ls::LinearSolver,
  A::AbstractParamMatrix,
  b::AbstractParamVector)

  x = allocate_in_domain(A)
  solve!(x,ls,A,b)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  A::AbstractParamMatrix,
  b::AbstractParamVector)

  A_item = testitem(A)
  ss = symbolic_setup(ls,A_item)
  ns = numerical_setup(ss,A_item)

  @inbounds for i in param_eachindex(x)
    xi = param_getindex(x,i)
    bi = param_getindex(b,i)
    solve!(xi,ns,bi)
    i == param_length(x) && continue
    Ai = param_getindex(A,i+1)
    numerical_setup!(ns,Ai)
  end

  ns
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  op::NonlinearOperator,
  cache::Nothing)

  fill!(x,zero(eltype(x)))
  b = residual(op,x)
  rmul!(b,-1)
  A = jacobian(op,x)
  ns = solve!(x,ls,A,b)

  Algebra.LinearSolverCache(A,b,ns)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  op::NonlinearOperator,
  cache::Algebra.LinearSolverCache)

  fill!(x,zero(eltype(x)))
  b = cache.b
  A = cache.A
  ns = cache.ns
  residual!(b,op,x)
  rmul!(b,-1)
  ns = solve!(x,ls,A,b)
  cache
end

# nonlinear interface

function Algebra.solve!(
  x::AbstractParamVector,
  nls::NewtonSolver,
  op::NonlinearOperator,
  cache::Nothing)

  b = residual(op,x)
  A = jacobian(op,x)
  A_item = testitem(A)
  x_item = testitem(x)
  dx = allocate_in_domain(A_item)
  fill!(dx,zero(eltype(dx)))
  ss = symbolic_setup(nls.ls,A_item)
  ns = numerical_setup(ss,A_item,x_item)
  Algebra._solve_nr!(x,A,b,dx,ns,nls,op)
  return NonlinearSolvers.NewtonCache(A,b,dx,ns)
end

function Algebra.solve!(
  x::AbstractParamVector,
  nls::NewtonSolver,
  op::NonlinearOperator,
  cache::NonlinearSolvers.NewtonCache)

  A,b,dx,ns = cache.A,cache.b,cache.dx,cache.ns
  residual!(b,op,x)
  jacobian!(A,op,x)
  Algebra._solve_nr!(x,A,b,dx,ns,nls,op)
  return cache
end

function Algebra._solve_nr!(x::AbstractParamVector,A,b,dx,ns,nls,op)
  log = nls.log

  res = norm(b)
  done = LinearSolvers.init!(log,res)

  while !done
    rmul!(b,-1)

    @inbounds for i in param_eachindex(x)
      xi = param_getindex(x,i)
      Ai = param_getindex(A,i)
      bi = param_getindex(b,i)
      numerical_setup!(ns,Ai)
      solve!(dx,ns,bi)
      xi .+= dx
    end

    residual!(b,op,x)
    res  = norm(b)
    done = LinearSolvers.update!(log,res)

    if !done
      jacobian!(A,op,x)
    end
  end

  LinearSolvers.finalize!(log,res)
  return x
end
