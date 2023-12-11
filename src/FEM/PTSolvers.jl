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
  θ = fesolver.θ
  dt = fesolver.dt
  t0 = fesolver.t0
  tf = fesolver.tf
  collect(t0:dt:tf-dt) .+ dt*θ
end

function TransientFETools.get_algebraic_operator(
  fesolver::PODESolver,
  feop::PTFEOperator,
  sols::PTArray,
  params::Table)

  dtθ = fesolver.θ == 0.0 ? fesolver.dt : fesolver.dt*fesolver.θ
  times = get_times(fesolver)
  ode_cache = allocate_cache(feop,params,times)
  ode_cache = update_cache!(ode_cache,feop,params,times)
  sols_cache = zero(sols)
  get_algebraic_operator(feop,params,times,dtθ,sols,ode_cache,sols_cache)
end

struct PODESolution
  solver::PODESolver
  op::PTFEOperator
  μ::AbstractVector
  u0::PTArray
  t0::Real
  tf::Real
end

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

function Algebra.symbolic_setup(s::BackslashSolver,mat::PTArray{<:AbstractMatrix})
  symbolic_setup(s,testitem(mat))
end

function Algebra.symbolic_setup(s::LUSolver,mat::PTArray{<:AbstractMatrix})
  symbolic_setup(s,testitem(mat))
end

function Algebra.numerical_setup(ss::Algebra.LUSymbolicSetup,mat::PTArray)
  ns = Vector{Algebra.LUNumericalSetup}(undef,length(mat))
  @inbounds for k = eachindex(mat)
    ns[k] = numerical_setup(ss,mat[k])
  end
  PTArray(ns)
end

function Algebra.numerical_setup!(ns,mat::PTArray)
  @inbounds for k = eachindex(mat)
    ns[k].factors = lu(mat[k])
  end
end

function Algebra.solve!(x::PTArray,ns,b::PTArray)
  @inbounds for k in eachindex(x)
    solve!(x[k],ns[k],b[k])
  end
end

struct PTAffineOperator <: NonlinearOperator
  matrix::PTArray
  vector::PTArray
end

function Algebra.solve!(x::PTArray,ls::LinearSolver,op::PTAffineOperator,::Nothing)
  A,b = op.matrix,op.vector
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A)
  solve!(x,ns,b)
  ns
end

function Algebra.solve!(x::PTArray,::LinearSolver,op::PTAffineOperator,ns)
  A,b = op.matrix,op.vector
  numerical_setup!(ns,A)
  solve!(x,ns,b)
  ns
end

struct PTLinearSolverCache <: GridapType
  A::PTArray{<:AbstractMatrix}
  b::PTArray{<:AbstractVector}
  ns::PTArray{<:NumericalSetup}
end

function Algebra.solve!(
  x::PTArray{<:AbstractVector},
  ls::LinearSolver,
  op::NonlinearOperator,
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
  x::PTArray{<:AbstractVector},
  ls::LinearSolver,
  op::NonlinearOperator,
  cache::PTLinearSolverCache)

  b = cache.b
  A = cache.A
  ns = cache.ns
  residual!(b,op,x)
  numerical_setup!(ns,A)
  rmul!(b,-1)
  solve!(x,ns,b)
  cache
end

function Algebra.LinearSolverCache(A::PTArray,b::PTArray,ns::PTArray)
  PTLinearSolverCache(A,b,ns)
end

struct PTNewtonRaphsonCache <: GridapType
  A::PTArray{<:AbstractMatrix}
  b::PTArray{<:AbstractVector}
  dx::PTArray{<:AbstractVector}
  ns::PTArray{<:NumericalSetup}
end

function Algebra.NewtonRaphsonCache(A::PTArray,b::PTArray,dx::PTArray,ns::PTArray)
  PTNewtonRaphsonCache(A,b,dx,ns)
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

function Algebra._solve_nr!(x::PTArray,A::PTArray,b::PTArray,dx::PTArray,ns,nls,op)
  _,conv0 = Algebra._check_convergence(nls.tol,b)
  for iter in 1:nls.max_nliters
    b .*= -1
    solve!(dx,ns,b)
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
