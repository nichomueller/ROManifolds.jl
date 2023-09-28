abstract type PODESolver <: ODESolver end

struct PThetaMethod <: PODESolver
  nls::NonlinearSolver
  uh0::Function
  θ::Float
  dt::Float
  t0::Real
  tf::Real
end

function get_time_ndofs(fesolver::PThetaMethod)
  dt = fesolver.dt
  t0 = fesolver.t0
  tf = fesolver.tf
  Int((tf-t0)/dt) - 1
end

function get_times(fesolver::PThetaMethod)
  θ = fesolver.θ
  dt = fesolver.dt
  t0 = fesolver.t0
  tf = fesolver.tf
  collect(t0:dt:tf-dt) .+ dt*θ
end

struct PODESolution{T}
  solver::PODESolver
  op::PODEOperator
  μ::AbstractVector
  u0::PTArray
  t0::Real
  tf::Real

  function PODESolution(
    solver::PODESolver,
    op::PODEOperator,
    μ::AbstractVector,
    u0::PTArray,
    t0::Real,
    tf::Real)

    test = get_test(op.feop)
    T = typeof(test)
    new{T}(solver,op,μ,u0,t0,tf)
  end
end

function Base.iterate(sol::PODESolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0
  n = 0
  cache = nothing

  uf,tf,cache = solution_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,tf,n),state
end

function Base.iterate(sol::PODESolution,state)
  uf,u0,t0,n,cache = state

  if t0 >= sol.tf - 100*eps()
    return nothing
  end

  uf,tf,cache = solution_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,tf,n),state
end

function collect_solutions(
  fesolver::PODESolver,
  feop::PTFEOperator,
  μ::Table)

  uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
  ode_op = get_algebraic_operator(feop)
  u0 = get_free_dof_values(uh0(μ))
  uμt = PODESolution(fesolver,ode_op,μ,u0,t0,tf)
  num_iter = Int(tf/fesolver.dt)
  sols = allocate_solution(ode_op,num_iter)
  for (u,t,n) in uμt
    printstyled("Computing fe solution at time $t for every parameter\n";color=:blue)
    sols[n] = get_solution(ode_op,u)
  end
  return Snapshots(sols)
end

for f in (:allocate_solution,:get_solution)
  @eval begin
    function $f(op::PODEOperator,args...)
      $f(get_test(op.feop),args...)
    end
  end
end

function allocate_solution(fe::FESpace,niter::Int)
  T = get_vector_type(fe)
  Vector{PTArray{T}}(undef,niter)
end

function allocate_solution(fe::MultiFieldFESpace,niter::Int)
  T = get_vector_type(fe)
  Vector{Vector{PTArray{T}}}(undef,niter)
end

function get_solution(::FESpace,free_values::PTArray)
  free_values
end

function get_solution(fe::MultiFieldFESpace,free_values::PTArray)
  blocks = map(1:length(fe.spaces)) do i
    restrict_to_field(fe,free_values,i)
  end
  PTArray(blocks)
end

function collect_residuals(
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table,
  args...;
  kwargs...)

  times = get_times(fesolver)
  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,times)
  b = allocate_residual(ode_op,sols,ode_cache)
  collect_residuals!(b,fesolver,ode_op,sols,μ,ode_cache,args...;kwargs...)
end

function collect_jacobians(
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table,
  args...;
  kwargs...)

  times = get_times(fesolver)
  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,times)
  A = allocate_jacobian(ode_op,sols,ode_cache)
  collect_jacobians!(A,fesolver,ode_op,sols,μ,ode_cache,args...;kwargs...)
end

for fun in (:collect_residuals!,:collect_jacobians!)
  @eval begin
    function $fun(
      q::PTArray,
      fesolver::PThetaMethod,
      ode_op::PODEOperator,
      sols::PTArray,
      μ::Table,
      ode_cache,
      args...;
      return_trian=false,
      kwargs...)

      dt,θ = fesolver.dt,fesolver.θ
      dtθ = θ == 0.0 ? dt : dt*θ
      times = get_times(fesolver)
      ode_cache = update_cache!(ode_cache,ode_op,μ,times)
      nlop = PThetaMethodNonlinearOperator(ode_op,μ,times,dtθ,sols,ode_cache,sols)
      $fun(Val(return_trian),q,nlop,sols,args...;kwargs...)
    end
  end
end

function collect_residuals!(
  return_trian,
  b::PTArray,
  nlop::PNonlinearOperator,
  sols::PTArray,
  args...)

  printstyled("Computing fe residual for every time and parameter\n";color=:blue)
  ress = residual!(b,nlop,sols,args...)
  return NnzMatrix(ress;nparams=length(nlop.μ))
end

function collect_residuals!(
  return_trian::Val{true},
  b::PTArray,
  nlop::PNonlinearOperator,
  sols::PTArray,
  args...)

  printstyled("Computing fe residual for every time and parameter\n";color=:blue)
  ress,trian = residual!(b,nlop,sols,return_trian)
  return NnzMatrix.(ress;nparams=length(nlop.μ)),trian
end

function collect_jacobians!(
  return_trian,
  A::PTArray,
  nlop::PNonlinearOperator,
  sols::PTArray,
  args...;
  i=1)

  printstyled("Computing fe jacobian #$i for every time and parameter\n";color=:blue)
  jacs_i = jacobian!(A,nlop,sols,i,args...)
  return NnzMatrix(map(NnzVector,jacs_i);nparams=length(nlop.μ))
end

function collect_jacobians!(
  return_trian::Val{true},
  A::PTArray,
  nlop::PNonlinearOperator,
  sols::PTArray,
  args...;
  i=1)

  printstyled("Computing fe residual for every time and parameter\n";color=:blue)
  jacs_i,trian = jacobian!(A,nlop,sols,i,return_trian)
  return map(x->NnzMatrix(map(NnzVector,x)),jacs_i),trian
end
