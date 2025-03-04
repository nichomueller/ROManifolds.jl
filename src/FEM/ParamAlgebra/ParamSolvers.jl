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
  x_item = testitem(x)
  ss = symbolic_setup(ls,A_item)
  ns = numerical_setup(ss,A_item,x_item)
  solve!(x,ns,A,b)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ns::NumericalSetup,
  A::AbstractParamMatrix,
  b::AbstractParamVector)

  @inbounds for i in param_eachindex(x)
    Ai = param_getindex(A,i)
    xi = param_getindex(x,i)
    bi = param_getindex(b,i)
    rmul!(bi,-1)
    numerical_setup!(ns,Ai)
    solve!(xi,ns,bi)
  end

  ns
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  op::NonlinearOperator,
  cache::Nothing)

  b = allocate_residual(op,x)
  A = allocate_jacobian(op,x)
  cache = SystemCache(A,b)
  solve!(x,ls,op,cache)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  op::NonlinearOperator,
  cache::SystemCache)

  fill!(x,zero(eltype(x)))
  @unpack A,b = cache
  residual!(b,op,x)
  jacobian!(A,op,x)
  ns = solve!(x,ls,A,b)
  Algebra.LinearSolverCache(A,b,ns)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  op::NonlinearOperator,
  cache::Algebra.LinearSolverCache)

  fill!(x,zero(eltype(x)))
  @unpack A,b,ns = cache
  residual!(b,op,x)
  jacobian!(A,op,x)
  solve!(x,ns,A,b)
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

  @unpack A,b,dx,ns = cache
  residual!(b,op,x)
  jacobian!(A,op,x)
  Algebra._solve_nr!(x,A,b,dx,ns,nls,op)
  return cache
end

function Algebra._solve_nr!(
  x::AbstractParamVector,
  A::AbstractParamMatrix,
  b::AbstractParamVector,
  dx::AbstractParamVector,
  ns,nls,op)

  log = nls.log

  res = norm(b)
  done = LinearSolvers.init!(log,res)

  while !done
    @inbounds for i in param_eachindex(x)
      xi = param_getindex(x,i)
      Ai = param_getindex(A,i)
      bi = param_getindex(b,i)
      numerical_setup!(ns,Ai)
      rmul!(bi,-1)
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

# lazy iteration over the parameters

struct ParamSolver{S<:NonlinearSolver} <: NonlinearSolver
  solver::S
end

Algebra.symbolic_setup(nls::ParamSolver,A::AbstractMatrix) = symbolic_setup(nls.solver,A)

ParamSolver() = ParamSolver(FESolver())

const LinearParamSolver = ParamSolver{<:LinearSolver}

function _lazy_solve!(
  x::AbstractParamVector,
  ls::LinearParamSolver,
  op::NonlinearOperator,
  A::AbstractMatrix,
  b::AbstractVector)

  x_item = testitem(x)
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A,x_item)
  _lazy_solve!(x,A,b,ns,ls,op)
end

function _lazy_solve!(x,A,b,ns,ls,op)
  reset_index!(op)
  @inbounds for i in param_eachindex(x)
    xi = param_getindex(x,i)
    residual!(b,op,x)
    jacobian!(A,op,x)
    rmul!(b,-1)
    numerical_setup!(ns,A)
    solve!(xi,ns,b)
    next_index!(op)
  end
  empty_matvecdata!(op)
  return ns
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearParamSolver,
  op::NonlinearOperator,
  cache::Nothing)

  b = allocate_residual(op,x)
  A = allocate_jacobian(op,x)
  cache = SystemCache(A,b)
  solve!(x,ls,op,cache)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearParamSolver,
  op::NonlinearOperator,
  cache::SystemCache)

  @unpack A,b = cache
  residual!(b,op,x)
  jacobian!(A,op,x)
  _lazy_solve!(x,ls,op,A,b)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearParamSolver,
  op::NonlinearOperator,
  cache::Algebra.LinearSolverCache)

  @unpack A,b,ns = cache
  residual!(b,op,x)
  jacobian!(A,op,x)
  _lazy_solve!(x,A,b,ns,ls,op)
  cache
end

const NonlinearParamSolver = ParamSolver{<:NonlinearSolver}

function Algebra.solve!(
  x::AbstractParamVector,
  solver::NonlinearParamSolver,
  op::NonlinearOperator,
  cache::NonlinearSolvers.NewtonCache)

  @unpack A,b,dx,ns = cache
  nls = solver.nls
  reset_index!(op)
  lazy_residual!(b,op,x)
  lazy_jacobian!(A,op,x)
  @inbounds for i in param_eachindex(x)
    _lazy_solve_nr!(x,A,b,dx,ns,nls,op,i)
    next_index!(op)
  end
  empty_matvecdata!(op)
  return cache
end

function _lazy_solve_nr!(x,A,b,dx,ns,nls,op,i)
  log = nls.log

  res = norm(b)
  done = LinearSolvers.init!(log,res)

  xi = param_getindex(x,i)

  while !done
    rmul!(b,-1)
    solve!(dx,ns,b)
    xi .+= dx

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
