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
  Int((tf-t0)/dt)
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

  uf,tf,cache = solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

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

  uf,tf,cache = solve_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,tf,n),state
end

function collect_solutions(
  fesolver::PODESolver,
  feop::PTFEOperator,
  trial::PTTrialFESpace,
  μ::Table)

  uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
  trial_μt = trial(μ,t0)
  ode_op = get_algebraic_operator(feop)
  u0 = get_free_dof_values(uh0(μ))
  time_ndofs = get_time_ndofs(fesolver)
  T = get_vector_type(trial_μt)
  uμt = PODESolution(fesolver,ode_op,μ,u0,t0,tf)
  sols = Vector{PTArray{T}}(undef,time_ndofs)
  stats = @timed for (sol,t,n) in uμt
    println("Computing fe solution at time $t for every parameter")
    sols[n] = copy(sol)
  end
  return Snapshots(sols),stats
end

function collect_solutions(
  fesolver::PODESolver,
  feop::PTFEOperator,
  trial::PTMultiFieldTrialFESpace,
  μ::Table)

  uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
  trial_μt = trial(μ,t0)
  ode_op = get_algebraic_operator(feop)
  u0 = get_free_dof_values(uh0(μ))
  time_ndofs = get_time_ndofs(fesolver)
  T = get_vector_type(trial_μt)
  uμt = PODESolution(fesolver,ode_op,μ,u0,t0,tf)
  sols = Vector{Vector{PTArray{T}}}(undef,time_ndofs)
  stats = @timed for (sol,t,n) in uμt
    println("Computing fe solution at time $t for every parameter")
    sols[n] = split_fields(trial_μt,copy(sol))
  end
  return BlockSnapshots(sols),stats
end

for fun in (:collect_residuals_for_idx!,:collect_jacobians_for_idx!)
  @eval begin
    function $fun(
      q::PTArray,
      fesolver::PThetaMethod,
      feop::PTFEOperator,
      sols::PTArray,
      μ::Table,
      times::Vector{<:Real},
      args...;
      kwargs...)

      dt,θ = fesolver.dt,fesolver.θ
      dtθ = θ == 0.0 ? dt : dt*θ
      ode_op = get_algebraic_operator(feop)
      ode_cache = allocate_cache(ode_op,μ,times)
      ode_cache = update_cache!(ode_cache,ode_op,μ,times)
      sols_cache = copy(sols) .* 0.
      nlop = get_nonlinear_operator(ode_op,μ,times,dtθ,sols,ode_cache,sols_cache)
      $fun(q,nlop,sols,args...;kwargs...)
    end
  end
end

function collect_residuals_for_trian(
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table,
  times::Vector{<:Real},
  args...;
  kwargs...)

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,times)
  b = allocate_residual(ode_op,μ,times,sols,ode_cache)
  collect_residuals_for_trian!(b,fesolver,ode_op,sols,μ,times,ode_cache,args...;kwargs...)
end

function collect_jacobians_for_trian(
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table,
  times::Vector{<:Real},
  args...;
  kwargs...)

  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,times)
  A = allocate_jacobian(ode_op,μ,times,sols,ode_cache)
  collect_jacobians_for_trian!(A,fesolver,ode_op,sols,μ,times,ode_cache,args...;kwargs...)
end

for fun in (:collect_residuals_for_trian!,:collect_jacobians_for_trian!)
  @eval begin
    function $fun(
      q::PTArray,
      fesolver::PThetaMethod,
      ode_op::PODEOperator,
      sols::PTArray,
      μ::Table,
      times::Vector{<:Real},
      ode_cache,
      args...;
      kwargs...)

      dt,θ = fesolver.dt,fesolver.θ
      dtθ = θ == 0.0 ? dt : dt*θ
      ode_cache = update_cache!(ode_cache,ode_op,μ,times)
      sols_cache = copy(sols) .* 0.
      nlop = get_nonlinear_operator(ode_op,μ,times,dtθ,sols,ode_cache,sols_cache)
      $fun(q,nlop,sols,args...;kwargs...)
    end
  end
end

function collect_residuals_for_idx!(
  b::PTArray,
  nlop::PNonlinearOperator,
  sols::PTArray,
  nonzero_idx::Vector{Int},
  args...)

  println("Computing fe residual for every time and parameter")
  ress = residual!(b,nlop,sols,args...)
  return hcat(map(x->getindex(x,nonzero_idx),ress.array)...)
end

function collect_residuals_for_trian!(
  b::PTArray,
  nlop::PNonlinearOperator,
  sols::PTArray,
  args...)

  println("Computing fe residual for every time and parameter")
  ress,trian = residual_for_trian!(b,nlop,sols)
  return NnzMatrix.(ress;nparams=length(nlop.μ)),trian
end

function collect_jacobians_for_idx!(
  A::PTArray,
  nlop::PNonlinearOperator,
  sols::PTArray,
  nonzero_idx::Vector{Int},
  args...;
  i=1)

  println("Computing fe jacobian #$i for every time and parameter")
  jacs_i = jacobian!(A,nlop,sols,i,args...)
  return Matrix(hcat(map(x->x[nonzero_idx],jacs_i.array)...))
end

function collect_jacobians_for_trian!(
  A::PTArray,
  nlop::PNonlinearOperator,
  sols::PTArray,
  args...;
  i=1)

  println("Computing fe jacobian #$i for every time and parameter")
  jacs_i,trian = jacobian_for_trian!(A,nlop,sols,i)
  return map(x->NnzMatrix(map(NnzVector,x);nparams=length(nlop.μ)),jacs_i),trian
end
