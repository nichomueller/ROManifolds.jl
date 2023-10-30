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

function recenter(a::PTArray,ah0::PTFEFunction;θ::Real=1)
  a0 = get_free_dof_values(ah0)
  recenter(a,a0;θ)
end

function recenter(a::Vector{<:PTArray},ah0::PTFEFunction;θ::Real=1)
  map(eachindex(a)) do i
    ai = a[i]
    ai0 = get_free_dof_values(ah0[i])
    recenter(ai,ai0;θ)
  end
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

function collect_single_field_solutions(
  fesolver::PODESolver,
  feop::PTFEOperator,
  μ::Table)

  uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
  ode_op = get_algebraic_operator(feop)
  u0 = get_free_dof_values(uh0(μ))
  time_ndofs = get_time_ndofs(fesolver)
  nparams = length(μ)
  T = get_vector_type(feop.test)
  uμt = PODESolution(fesolver,ode_op,μ,u0,t0,tf)
  sols = Vector{PTArray{T}}(undef,time_ndofs)
  println("Computing fe solution: time marching across $time_ndofs instants, for $nparams parameters")
  stats = @timed for (sol,n) in uμt
    sols[n] = copy(sol)
  end
  println("Time marching complete")
  return Snapshots(sols),ComputationInfo(stats,nparams)
end

function collect_multi_field_solutions(
  fesolver::PODESolver,
  feop::PTFEOperator,
  μ::Table)

  uh0,t0,tf = fesolver.uh0,fesolver.t0,fesolver.tf
  ode_op = get_algebraic_operator(feop)
  u0 = get_free_dof_values(uh0(μ))
  time_ndofs = get_time_ndofs(fesolver)
  nparams = length(μ)
  T = get_vector_type(feop.test)
  uμt = PODESolution(fesolver,ode_op,μ,u0,t0,tf)
  sols = Vector{Vector{PTArray{T}}}(undef,time_ndofs)
  println("Computing fe solution: time marching across $time_ndofs instants, for $nparams parameters")
  stats = @timed for (sol,n) in uμt
    sols[n] = split_fields(feop.test,copy(sol))
  end
  println("Time marching complete")
  return BlockSnapshots(sols),ComputationInfo(stats,nparams)
end

function init_collector(
  fesolver::PThetaMethod,
  feop::PTFEOperator,
  sols::PTArray,
  μ::Table,
  times::Vector{<:Real})

  dtθ = fesolver.θ == 0.0 ? fesolver.dt : fesolver.dt*fesolver.θ
  ode_op = get_algebraic_operator(feop)
  ode_cache = allocate_cache(ode_op,μ,times)
  ode_cache = update_cache!(ode_cache,ode_op,μ,times)
  sols_cache = zero(sols)
  nlop = get_nonlinear_operator(ode_op,μ,times,dtθ,sols,ode_cache,sols_cache)
  return nlop
end

function collect_residuals_for_trian(nlop::PNonlinearOperator)
  b = allocate_residual(nlop,nlop.u0)
  ress,trian = residual_for_trian!(b,nlop,nlop.u0)
  return NnzMatrix.(ress;nparams=length(nlop.μ)),trian
end

function collect_jacobians_for_trian(nlop::PNonlinearOperator;i=1)
  A = allocate_jacobian(nlop,nlop.u0)
  jacs_i,trian = jacobian_for_trian!(A,nlop,nlop.u0,i)
  return map(x->NnzMatrix(map(NnzVector,x);nparams=length(nlop.μ)),jacs_i),trian
end

function collect_residuals_for_idx!(
  cache,
  sols::PTArray{T},
  nonzero_idx::Vector{Int},
  meas::Measure) where T

  b,nlop = cache
  ress = residual_for_idx!(b,nlop,sols,meas)
  N = length(ress)
  resmat = zeros(eltype(T),length(nonzero_idx),N)
  @inbounds for n = 1:N
    resmat[:,n] = ress[n][nonzero_idx]
  end
  return resmat
end

function collect_jacobians_for_idx!(
  cache,
  sols::PTArray{T},
  nonzero_idx::Vector{Int},
  meas::Measure;
  i=1) where T

  A,nlop = cache
  jacs_i = jacobian_for_idx!(A,nlop,sols,i,meas)
  N = length(jacs_i)
  jacimat = zeros(eltype(T),length(nonzero_idx),N)
  @inbounds for n = 1:N
    jacimat[:,n] = jacs_i[n][nonzero_idx].nzval
  end
  return jacimat
end
