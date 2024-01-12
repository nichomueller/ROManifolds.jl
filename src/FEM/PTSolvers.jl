abstract type PODESolver <: ODESolver end

struct PThetaMethod <: PODESolver
  nls::NonlinearSolver
  uh0::Function
  θ::Float
  dt::Float
  t0::Real
  tf::Real
end

function num_time_dofs(fesolver::PThetaMethod)
  dt = fesolver.dt
  t0 = fesolver.t0
  tf = fesolver.tf
  floor(Int,(tf-t0)/dt)
end

function get_times(fesolver::PThetaMethod)
  dt = fesolver.dt
  t0 = fesolver.t0
  tf = fesolver.tf
  collect(t0:dt:tf-dt)
end

function get_stencil_times(fesolver::PThetaMethod)
  θ = fesolver.θ
  dt = fesolver.dt
  get_times(fesolver) .+ dt*θ
end

function TransientFETools.get_algebraic_operator(
  fesolver::PODESolver,
  feop::TransientPFEOperator,
  sols::PTArray,
  params::Table)

  dtθ = fesolver.θ == 0.0 ? fesolver.dt : fesolver.dt*fesolver.θ
  times = get_stencil_times(fesolver)
  ode_cache = allocate_cache(feop,params,times)
  ode_cache = update_cache!(ode_cache,feop,params,times)
  sols_cache = zero(sols)
  get_algebraic_operator(feop,params,times,dtθ,sols,ode_cache,sols_cache)
end

struct PODESolution
  solver::PODESolver
  op::TransientPFEOperator
  μ::AbstractVector
  u0::AbstractVector
  t0::Real
  tf::Real
end

Base.length(sol::PODESolution) = Int((sol.tf-sol.t0)/sol.solver.dt)

function Base.iterate(sol::PODESolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0
  n = 0
  cache = nothing

  uf,tf,cache = solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,n),state
end

function Base.iterate(sol::PODESolution,state)
  uf,u0,t0,n,cache = state

  if t0 >= sol.tf - 100*eps()
    return nothing
  end

  uf,tf,cache = solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,n),state
end

Algebra.symbolic_setup(s::BackslashSolver,mat::PTArray) = symbolic_setup(s,testitem(mat))

Algebra.symbolic_setup(s::BackslashSolver,mat::AbstractArray{<:PTArray}) = symbolic_setup(s,testitem(mat))

Algebra.symbolic_setup(s::LUSolver,mat::PTArray) = symbolic_setup(s,testitem(mat))

Algebra.symbolic_setup(s::LUSolver,mat::AbstractArray{<:PTArray}) = symbolic_setup(s,testitem(mat))

struct PTAffineOperator <: PTNonlinearOperator
  matrix::AbstractArray
  vector::AbstractVector
end

function Algebra.solve!(
  x::AbstractVector,
  ls::LinearSolver,
  op::PTAffineOperator,
  ::Nothing)

  A,b = op.matrix,op.vector
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A)
  solve!(x,ns,b)
  ns
end

function Algebra.solve!(
  x::AbstractVector,
  ::LinearSolver,
  op::PTAffineOperator,
  ns)

  A,b = op.matrix,op.vector
  numerical_setup!(ns,A)
  solve!(x,ns,b)
  ns
end

struct PTLinearSolverCache <: GridapType
  A::AbstractMatrix
  b::AbstractVector
  ns::NumericalSetup
end

function Algebra.solve!(
  x::AbstractVector,
  ls::LinearSolver,
  op::PTNonlinearOperator,
  cache::Nothing)

  b = residual(op,x)
  A = jacobian(op,x)
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A)
  rmul!(b,-1)
  solve!(x,ns,b)
  Algebra.LinearSolverCache(A,b,ns)
end

function Algebra.solve!(
  x::AbstractVector,
  ls::LinearSolver,
  op::PTNonlinearOperator,
  cache::PTLinearSolverCache)

  b = cache.b
  A = cache.A
  ns = cache.ns
  residual!(b,op,x)
  jacobian!(A,op,x)
  numerical_setup!(ns,A)
  rmul!(b,-1)
  solve!(x,ns,b)
  cache
end

function Algebra.LinearSolverCache(A::PTArray,b::PTArray,ns::NumericalSetup)
  PTLinearSolverCache(A,b,ns)
end

function Algebra._check_convergence(nls,b::PTArray,m0)
  m = maximum(abs,b)
  return all(m .< nls.tol * m0)
end
