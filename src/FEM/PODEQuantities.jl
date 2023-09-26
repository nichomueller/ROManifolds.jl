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
  solutions = allocate_solution(ode_op,num_iter)
  for (u,t,n) in uμt
    printstyled("Computing fe solution at time $t for every parameter\n";color=:blue)
    solutions[n] = get_solution(ode_op,u)
  end
  return solutions
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
  PTArray.(blocks)
end

function center_solution(
  solver::PThetaMethod,
  sols::Vector{<:PTArray{T}},
  μ::Table) where T

  uh0 = solver.uh0
  u0 = get_free_dof_values(uh0(μ))
  solsθ = sols*θ + [u0,sols[2:end]...]*(1-θ)
  to_ptarray(PTArray{T},solsθ)
end

function get_at_params(range::UnitRange,sols::Vector{<:PTArray{T}}) where T
  time_ndofs = length(sols)
  nparams = length(range)
  array = Vector{T}(undef,time_ndofs*nparams)
  for nt in eachindex(sols)
    for np in range
      array[(nt-1)*time_ndofs+np] = sols[nt][np]
    end
  end
  return PTArray(array)
end
