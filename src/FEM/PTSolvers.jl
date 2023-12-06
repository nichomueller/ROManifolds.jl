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
  Int((tf-t0)/dt)
end

function get_times(fesolver::PThetaMethod)
  θ = fesolver.θ
  dt = fesolver.dt
  t0 = fesolver.t0
  tf = fesolver.tf
  collect(t0:dt:tf-dt) .+ dt*θ
end

function get_ptoperator(
  fesolver::PODESolver,
  feop::PTFEOperator,
  sols::PTArray,
  params::Table)

  dtθ = fesolver.θ == 0.0 ? fesolver.dt : fesolver.dt*fesolver.θ
  ode_op = get_algebraic_operator(feop)
  times = get_times(fesolver)
  ode_cache = allocate_cache(ode_op,params,times)
  ode_cache = update_cache!(ode_cache,ode_op,params,times)
  sols_cache = zero(sols)
  get_ptoperator(ode_op,params,times,dtθ,sols,ode_cache,sols_cache)
end

struct PODESolution
  solver::PODESolver
  op::PODEOperator
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

function Algebra.numerical_setup(ss::Algebra.LUSymbolicSetup,mat::PTArray)
  ns = Vector{Algebra.LUNumericalSetup}(undef,length(mat))
  @inbounds for k = eachindex(mat)
    ns[k] = numerical_setup(ss,mat[k])
  end
  ns
end

function Algebra.numerical_setup!(ns,mat::PTArray)
  @inbounds for k = eachindex(mat)
    ns[k].factors = lu(mat[k])
  end
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

function Algebra.solve!(x::PTArray,ls::LinearSolver,op::PTAffineOperator,::Nothing)
  A,b = op.matrix,op.vector
  @assert length(A) == length(b) == length(x)
  ss = symbolic_setup(ls,testitem(A))
  ns = numerical_setup(ss,A)
  _loop_solve!(x,ns,b)
  ns
end

function Algebra.solve!(x::PTArray,::LinearSolver,op::PTAffineOperator,ns)
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

function Algebra.solve!(
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

function Algebra.solve!(
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
    b .*= -1
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
